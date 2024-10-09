import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import yaml
import argparse
from accelerate import Accelerator
from utils.losses import L1loss, FFTloss
from utils.logger import build_logger
from BaseTrainer import BaseTrainer
from utils.metric_utils import calculate_psnr
import os
import shutil

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, gelu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel,
                                             kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if gelu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class depthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depthWiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            depthWiseConv(in_channel, out_channel),
            nn.GELU(),
            depthWiseConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel)
                  for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DMFF(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.conv1 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, y):
        # fusion propagation
        feat_fusion = torch.cat([x, y], dim=1)  # b 128 256 256
        feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
        feat_prop1, feat_prop2 = torch.split(feat_fusion, self.n_feat, dim=1)
        feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
        feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
        x = feat_prop1 + feat_prop2
        return x

class Net(nn.Module):
    def __init__(self, num_res=[4,12,16], base_channel=32):
        super().__init__()

        # stage_1
        self.stage1_convIn = BasicConv(3, base_channel, kernel_size=3, gelu=True, stride=1)
        self.stage1_encoder = EBlock(base_channel, num_res[0])

        # stage_2
        self.stage2_SCM = nn.Conv2d(3, base_channel * 2, kernel_size=3, stride=1, padding=1)
        self.stage2_convIn = BasicConv(base_channel, base_channel * 2, kernel_size=3, gelu=True, stride=2)
        self.stage2_atb = DMFF(base_channel * 2)
        self.stage2_encoder = EBlock(base_channel * 2, num_res[1])

        # stage_3
        self.stage3_SCM = nn.Conv2d(3, base_channel * 4, kernel_size=3, stride=1, padding=1)
        self.stage3_convIn = BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, gelu=True, stride=2)
        self.stage3_atb = DMFF(base_channel * 4)
        self.stage3_encoder = EBlock(base_channel * 4, num_res[2])

        # stage_3
        self.stage3_decoder = DBlock(base_channel * 4, num_res[2])
        self.stage3_transpose = BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, gelu=True, stride=2,
                                          transpose=True)
        self.stage3_convOut = BasicConv(base_channel * 4, 3, kernel_size=3, gelu=False, stride=1)

        # stage_2
        self.stage2_feat_extract = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, gelu=True, stride=1)
        self.stage2_decoder = DBlock(base_channel * 2, num_res[1])
        self.stage2_transpose = BasicConv(base_channel * 2, base_channel, kernel_size=4, gelu=True, stride=2,
                                          transpose=True)
        self.stage2_convOut = BasicConv(base_channel * 2, 3, kernel_size=3, gelu=False, stride=1)

        # stage_1
        self.stage1_feat_extract = BasicConv(base_channel * 2, base_channel, kernel_size=1, gelu=True, stride=1)
        self.stage1_decoder = DBlock(base_channel, num_res[0])
        self.stage1_convOut = BasicConv(base_channel, 3, kernel_size=3, gelu=False, stride=1)

    def forward(self, x):
        outputs = list()
        '''
        b, c, h, w = x.shape
        padsize = 16
        h_n = (padsize - h % padsize) % padsize
        w_n = (padsize - w % padsize) % padsize
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        '''
        stage2_x = F.interpolate(x, scale_factor=0.5)
        stage3_x = F.interpolate(stage2_x, scale_factor=0.5)
        stage2_z2 = self.stage2_SCM(stage2_x)
        stage3_z2 = self.stage3_SCM(stage3_x)

        # encoder
        stage1_x_shallow_feature = self.stage1_convIn(x)
        stage1_res = self.stage1_encoder(stage1_x_shallow_feature)

        stage2_z = self.stage2_convIn(stage1_res)
        stage2_z = self.stage2_atb(stage2_z, stage2_z2)
        stage2_res = self.stage2_encoder(stage2_z)

        stage3_z = self.stage3_convIn(stage2_res)
        stage3_z = self.stage3_atb(stage3_z, stage3_z2)
        stage3_res = self.stage3_encoder(stage3_z)

        # decoder
        stage3_out = self.stage3_decoder(stage3_res)
        stage3_out_ = self.stage3_transpose(stage3_out)
        stage3_out = self.stage3_convOut(stage3_out)
        outputs.append(stage3_x + stage3_out)

        stage2_out = torch.cat([stage3_out_, stage2_res], dim=1)
        stage2_out = self.stage2_feat_extract(stage2_out)
        stage2_out = self.stage2_decoder(stage2_out)
        stage2_out_ = self.stage2_transpose(stage2_out)
        stage2_out = self.stage2_convOut(stage2_out)
        outputs.append(stage2_out + stage2_x)

        stage1_out = torch.cat([stage2_out_, stage1_res], dim=1)
        stage1_out = self.stage1_feat_extract(stage1_out)
        stage1_out = self.stage1_decoder(stage1_out)
        stage1_out = self.stage1_convOut(stage1_out)
        res = stage1_out + x
        #res = res[:, :, :h, :w]
        outputs.append(res)

        return outputs

class Trainer(BaseTrainer):
    def __init__(self, opt):
        
        super().__init__(opt)
        self.net = Net(num_res=opt["net"]["num_res"],
                       base_channel=opt["net"]["base_channel"])
        
        self.init_net_params()
        self.init_optimizer()
        self.init_scheduler()
        self.init_dataset()
        self.init_logger()
        self.init_tensorboard()
        if opt["train"]["resume_path"]:
            self.resume_ckpt(opt["train"]["resume_path"])
            
        self.net, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.net, self.optimizer, self.lr_scheduler)
        self.train_dataset, self.val_dataset = self.accelerator.prepare(self.train_dataset, self.val_dataset)
    def train_step(self, lq, gt):
        gt_list = []
        gt_list.append(F.interpolate(gt, scale_factor=0.25))
        gt_list.append(F.interpolate(gt, scale_factor=0.5))
        gt_list.append(gt)

        loss_dict = {}
        total_loss = 0
        pixel_loss_value = 0
        fft_loss_value = 0
        preds = self.net(lq)

        for pred, gt in zip(preds, gt_list):
            if self.opt["train"]["L1loss"]:
                pixel_loss_value += L1loss(pred, gt, 
                                           weight=self.opt["train"]["L1loss"]["weight"],
                                           reduction=self.opt["train"]["L1loss"]["reduction"])
            if self.opt["train"]["FFTloss"]:
                fft_loss_value += FFTloss(pred, gt, 
                                          weight=self.opt["train"]["FFTloss"]["weight"],
                                          reduction=self.opt["train"]["FFTloss"]["reduction"])
        loss_dict["L1_loss"] = pixel_loss_value.item()
        loss_dict["FFT_loss"] = fft_loss_value.item()
        total_loss = pixel_loss_value + fft_loss_value
        self.accelerator.backward(total_loss)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_dict
    def train(self):
        log = build_logger(self.logger_name)
        finish_flag = False
        self.net.train()
        for epoch in range(self.start_epoch, self.total_epoch):
            for data in self.train_dataset:
                lq = data["lq"]
                gt = data["gt"]
                self.optimizer.zero_grad()
                loss_dict = self.train_step(lq, gt)
                self.record_loss(loss_dict)
                self.current_iter += 1
                # validation
                if self.current_iter % self.opt["val"]["val_freq"] == 0:
                    self.accelerator.wait_for_everyone()
                    self.validation()
                
                # save ckpt
                if self.current_iter % self.opt["val"]["save_freq"] == 0:
                    self.accelerator.wait_for_everyone()
                    save_ckpt_path = os.path.join(self.experiment_root, "weight")
                    os.makedirs(save_ckpt_path, exist_ok=True)
                    ckpt_name = f"net_{self.current_iter}.pth"
                    save_ckpt_path = os.path.join(save_ckpt_path, ckpt_name)
                    self.save_ckpt(save_ckpt_path)

                if self.accelerator.is_local_main_process:
                    if self.current_iter % self.opt["val"]["print_freq"] == 0:
                        msg = "Epoch: {}, Iters: {}/{}, Lr:{}, Loss:{}".format(epoch, self.current_iter, self.opt["train"]["iters"], self.get_current_learning_rate(), loss_dict)
                        log.info(msg)
                    if self.current_iter == self.opt["train"]["iters"]:
                        # self.accelerator.wait_for_everyone()
                        print("Training Finished!")
                        log.info("Training Finished!")
                        finish_flag = True
                        break
            
            if finish_flag and self.accelerator.is_local_main_process:
                return
    def validation(self):
        self.net.eval()
        log = build_logger(self.logger_name)
        psnr_list = []
        with torch.no_grad():
            for data in self.val_dataset:
                lq = data["lq"]
                gt = data["gt"]
                pred = self.net(lq)[2]
                psnr = calculate_psnr(pred, gt)
                psnr_list.append(psnr)
        aveg_psnr = np.array(psnr_list).mean()
        log.info(f"Iter:{self.current_iter}, PNSR:{aveg_psnr}")
        self.writer.add_scalar("val/psnr", aveg_psnr, self.current_iter)
        self.net.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    assert os.path.isfile(args.config_path), f"{args.config_path} not found!"
    with open(args.config_path, "r") as file:
        opt = yaml.safe_load(file)
    if args.debug:
        opt["val"]["save_freq"] = 8
        opt["val"]["val_freq"] = 8
        opt["val"]["print_freq"] = 1
        opt["name"] = "debug_"+opt["name"]
    ## 初始化实验存放路径
    experiment_root = os.path.join("experiment", opt["name"])
    os.makedirs(experiment_root, exist_ok=True)
    ## 将配置文件保存到实验文件夹下
    shutil.copy(args.config_path, experiment_root)
    shutil.copy("trainer.py", experiment_root)

    trainer = Trainer(opt)
    trainer.train()