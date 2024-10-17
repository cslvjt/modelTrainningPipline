import torch
def calculate_psnr(img1_list, img2_list):
    """
    img1: tensor: b c h w
    img2: tensor: b c h w
    """
    psnr_list = []
    assert img1_list.shape == img2_list.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    for img1, img2 in zip(img1_list, img2_list):
        mse = torch.mean((img1-img2)**2)
        if mse == 0:
            return float("inf")
        psnr = -10. * torch.log10(mse).detach().cpu()
        psnr_list.append(psnr)
    return sum(psnr_list) / len(psnr_list)
        