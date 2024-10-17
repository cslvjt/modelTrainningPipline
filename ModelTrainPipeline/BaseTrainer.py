import torch
import torch.nn.functional as F
import os
import math
import numpy as np
import random
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataloader.pairedImageDataloader import build_dataloader
from utils.logger import build_logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer():
    def __init__(self, opt):
        self.accelerator = Accelerator()
        self.opt = opt
        self.experiment_root = os.path.join("experiment", opt["name"])
        
        self.init_seed()

        self.logger_name = "Deblur"

        self.start_epoch = 0
        self.current_iter = 0
    def init_seed(self):
        set_seed(self.opt["seed"])
        torch.manual_seed(self.opt["seed"])
        torch.cuda.manual_seed(self.opt["seed"])
        torch.cuda.manual_seed_all(self.opt["seed"])
        np.random.seed(self.opt["seed"])
        random.seed(self.opt["seed"])
    def init_logger(self):
        opt = self.opt
        if self.accelerator.is_local_main_process:
            log_file = os.path.join(self.experiment_root, "logger_{}.txt".format(opt["name"]))
            log = build_logger(self.logger_name, log_file)
            log.info("Experiment name:{}".format(self.opt["name"]))
            log.info("Seed:{}".format(self.opt["seed"]))
            log.info("dataset:{}".format(self.opt["dataset"]))
            log.info("Network:{}".format(self.opt["net"]))
            log.info("Train:{}".format(self.opt["train"]))
            total_devices = self.accelerator.num_processes
            log.info(f"The number of GPU device is {total_devices}")
            log.info(f"The number of trained images: {len(self.train_dataset)}")
            log.info(f"The number of validation images: {len(self.val_dataset)}")
            log.info(f"Total Epoch: {self.total_epoch}")
    def init_net_params(self):
        self.params = self.net.parameters()
    def init_dataset(self):
        opt = self.opt
        # define dataloader
        self.train_dataset = build_dataloader(opt['dataset']["train"], phase="train", shuffle=True)
        self.val_dataset = build_dataloader(opt["dataset"]["validation"], phase="val", shuffle=False)
        # calculate epoch
        # attention total epoch is need to multiply the number of devices
        self.total_epoch = math.ceil(opt["train"]["iters"]/len(self.train_dataset))* self.accelerator.num_processes
    def save_ckpt(self, ckpt_path):
        state = {
            "model_state_dict": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iter": self.current_iter,
            "scheduler": self.lr_scheduler.state_dict(),
            "lr": self.get_current_learning_rate()
        }
        torch.save(state, ckpt_path)
    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr'] 
    def resume_ckpt(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.net.load_state_dict(state["model_state_dict"])
        self.current_iter = state['current_iter']
        self.lr_scheduler.load_state_dict(state["scheduler"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.lr = state["lr"]
        log = build_logger(self.logger_name)
        log.info(f"Resume from iters: {self.current_iter}")
    def init_optimizer(self):
        optimizer_name = self.opt["train"]["optimizer"]["type"]
        lr = self.opt["train"]["optimizer"]["lr"]
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.params, lr)
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.params, lr)
        else:
            raise NotImplementedError(f'optimizer {optimizer_name} is not supported yet.')
    def init_scheduler(self):
        scheduler_name = self.opt["train"]["scheduler"]["type"]
        if scheduler_name == "CosineAnnealingLR":
            # self.lr_scheduler = CosineAnnealingRestartLR(self.optimizer, 
            #                                              periods=self.opt["train"]["scheduler"]["periods"],
            #                                              restart_weights=self.opt["train"]["scheduler"]["restart_weights"],
            #                                              eta_min=self.opt["train"]["scheduler"]["eta_min"])
            self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                                  T_max=self.opt["train"]["iters"] * 2,
                                                  eta_min=self.opt["train"]["scheduler"]["eta_min"])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_name} is not implemented yet.')
    
    def init_tensorboard(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.experiment_root, "tb_logger"))
        self.writer.add_text("config", str(self.opt))
    
    def record_loss(self, loss_dict):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"loss/{key}", value, self.current_iter)