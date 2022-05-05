import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2

from numpy.linalg import norm

import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as TF


color_palette = [0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 0] + [100, 100, 100] * 252


def postprocessing_pred(pred: np.array) -> np.array:

    label_cnt, labels = cv2.connectedComponentsWithAlgorithm(pred, 8, cv2.CV_32S, cv2.CCL_GRANA)
    if label_cnt == 2:
        if labels[0, 0] == pred[0, 0]:
            pred = labels
        else:
            pred = 1 - labels
    else:
        max_cnt, max_label = 0, 0
        for i in range(label_cnt):
            mask = labels == i
            if pred[mask][0] == 0:
                continue
            cnt = len(mask.nonzero()[0])
            if cnt > max_cnt:
                max_cnt = cnt
                max_label = i
        pred = labels == max_label

    return pred.astype(np.uint8)


def calc_uncertainty(score):

    # seg shape: bs, obj_n, h, w
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    return uncertainty


def save_seg_mask(pred, seg_path, palette=color_palette):

    seg_img = Image.fromarray(pred)
    seg_img.putpalette(palette)
    seg_img.save(seg_path)


def add_overlay(img, mask, colors=color_palette, alpha=0.4, cscale=1):

    ids = np.unique(mask)
    img_overlay = img.copy()
    ones_np = np.ones(img.shape) * (1 - alpha)

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    for i in ids[1:]:

        canvas = img * alpha + ones_np * np.array(colors[i])[::-1]

        binary_mask = mask == i
        img_overlay[binary_mask] = canvas[binary_mask]

        contour = binary_dilation(binary_mask) ^ binary_mask
        img_overlay[contour, :] = 0

    return img_overlay


def save_overlay(img, mask, overlay_path, colors=[255, 0, 0], alpha=0.4, cscale=1):

    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_overlay = add_overlay(img, mask, colors, alpha, cscale)
    cv2.imwrite(overlay_path, img_overlay)


def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def normalize(x):
    return x / norm(x, ord=2, axis=1, keepdims=True)


def dist(p0, p1, axis):
    return norm(p0 - p1, ord=2, axis=axis)


def resize_img(img, out_size):
    h, w = img.shape[:2]

    if h > w:
        w_new = int(out_size * w / h)
        h_new = out_size
    else:
        h_new = int(out_size * h / w)
        w_new = out_size

    img = cv2.resize(img, (w_new, h_new))
    return img


def unify_features(features):
    output_size = features['f0'].shape[-2:]
    feature_tuple = tuple()

    for key, f in features.items():
        if key != 'f0':
            f = NF.interpolate(
                f,
                size=output_size, mode='bilinear', align_corners=False
            )
        feature_tuple += (f,)

    unified_feature = torch.cat(feature_tuple, dim=1)

    return unified_feature


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(NF.pad(inp, pad_array))

    return out_list, pad_array
