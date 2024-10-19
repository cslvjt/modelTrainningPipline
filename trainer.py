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

# define your network 
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        return x

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
    
    # define model optimizer step
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
    opt["name"] = args.name
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