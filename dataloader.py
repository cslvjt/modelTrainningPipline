import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from utils.img_utils import read_img, paired_random_crop, augment
import os.path as osp
import os


class PairedImageDataset(data.Dataset):

    def __init__(self, opt, phase="train") -> None:
        super().__init__()
        assert phase in ["train", "val"], f"{phase} Error!"
        self.opt = opt
        self.phase = phase
        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        self.gt_list = os.listdir(self.gt_folder)
        self.lq_list = os.listdir(self.lq_folder)

    def __getitem__(self, index):
        gt_path = osp.join(self.gt_folder, self.gt_list[index])
        lq_path = osp.join(self.lq_folder, self.lq_list[index])

        gt_img = read_img(gt_path)
        lq_img = read_img(lq_path)
        if self.phase == "train":
            gt_img, lq_img = paired_random_crop(gt_img, lq_img, self.opt["patch_size"], self.opt["scale"])
            gt_img, lq_img = augment([gt_img, lq_img], self.opt['use_hflip'], self.opt['use_rot'])
        # print(gt_path, gt_img.shape)
        # print(lq_path, lq_img.shape)
        gt_img = torch.from_numpy(gt_img.copy())
        lq_img = torch.from_numpy(lq_img.copy())
        return {"gt": gt_img, "lq": lq_img, "gt_name":self.gt_list[index]}
    
    def __len__(self):
        return len(self.gt_list)
        


def build_dataloader(opt, phase="train", shuffle = True):
    batch_size = opt.get("batch_size", 1)
    num_workers = opt.get("num_workers", 1)
    dataset = PairedImageDataset(opt, phase)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

