import os
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .WaterNet import WaterNetV1


def mask_background(img, prediction, device):
    instance_n, h, w = prediction['instances']._fields['pred_masks'].shape

    mask_all = torch.ones((h, w), dtype=torch.bool).to(device)
    for i in range(instance_n):
        mask_all = mask_all * torch.bitwise_not(prediction['instances']._fields['pred_masks'][i])

    mask_all = mask_all.unsqueeze(0).expand(3, -1, -1)
    zero_ele = torch.zeros(1).to(device)
    img = torch.where(mask_all, img, zero_ele)

    return img


def seg_water(img, prediction, device=torch.device('cuda')):
    water_net_v1 = WaterNetV1().to(device).eval()

    cp_path = '/Ship01/SourcesArchives/WaterNetV1/models/cp_WaterNet_199.pth.tar'
    if os.path.isfile(cp_path):
        print(f'Load checkpoint {cp_path}')
        checkpoint = torch.load(cp_path)
        water_net_v1.load_state_dict(checkpoint['water_net_v1'])

    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).to(device) / 255
    # img = TF.to_tensor(img).to(device)
    img = mask_background(img, prediction, device)
    water_img = img.cpu().numpy().transpose(1, 2, 0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = TF.normalize(img, mean, std).unsqueeze(0)

    seg_res = water_net_v1(img).detach()  # Size: (1, 1, h, w)
    water_mask = seg_res[0].cpu().numpy().transpose(1, 2, 0)

    return water_mask, water_img
