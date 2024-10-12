import torch
import os
from trainer import LWDNet
from utils.img_utils import read_img, save_img
from utils.metric_utils import calculate_psnr

def load_model(model_path):
    model = LWDNet(num_res=[8,12,16], base_channel=32)
    state_dict = torch.load(model_path, map_location="cuda:0")
    model.load_state_dict(state_dict["model_state_dict"])
    model.cuda()
    print(model_path)
    return model.eval()

def inference(dataset_root_path, save_root_path, model_path):
    root_lq = os.path.join(dataset_root_path, "blur")
    root_gt = os.path.join(dataset_root_path, "gt")
    img_list = os.listdir(root_lq)
    model = load_model(model_path)

    psnr_list = []
    for img_name in img_list:
        lq_path = os.path.join(root_lq, img_name)
        gt_path = os.path.join(root_gt, img_name)
        save_path = os.path.join(save_root_path, img_name)
        lq_img = read_img(lq_path)
        gt_img = read_img(gt_path)
        gt_img = torch.from_numpy(gt_img).unsqueeze(0) # b c h w
        with torch.no_grad():
            lq_img = torch.from_numpy(lq_img).unsqueeze(0).cuda()
            pred = model(lq_img)[2] # b c h w
        save_img(pred, save_path)
        psnr = calculate_psnr(pred.cpu(), gt_img).item()
        psnr_list.append(psnr)
        print(save_path, psnr)
    print("avg psnr:", sum(psnr_list) / len(psnr_list))

if __name__ == "__main__":
    root_path = "/data1/lvjiangtao/lvjiangtao/dataset/GoPro_train_latest/test"
    root_save_path = "/data1/lvjiangtao/ImageRestoration/LWDNet/experiment/LWDNet_V3_0917/visualization/LWDNet_60W"
    os.makedirs(root_save_path, exist_ok=True)
    model_path = "/data1/lvjiangtao/ImageRestoration/LWDNet/experiment/LWDNet_V3_0917/weight/net_600000.pth"
    inference(root_path, root_save_path, model_path)

