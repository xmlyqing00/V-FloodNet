import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2
from numpy.linalg import norm

import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as TF



def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def normalize(x):
    return x / norm(x, ord=2, axis=1, keepdims=True)


def dist(p0, p1, axis):
    return norm(p0 - p1, ord=2, axis=axis)


def add_overlay(img, mask, colors, alpha=0.7, cscale=1):
    ids = np.unique(mask)
    img_overlay = img.copy()
    ones_np = np.ones(img.shape) * (1 - alpha)

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    for i in ids[1:]:
        canvas = img * alpha + ones_np * np.array(colors[i])[::-1]  # to BGR order

        binary_mask = mask == i
        img_overlay[binary_mask] = canvas[binary_mask]

        contour = binary_dilation(binary_mask) ^ binary_mask
        img_overlay[contour, :] = 0

    return img_overlay


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
