import random
import math
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_erosion, binary_dilation
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop

random_thres = 0.8


def random_adjust_color(img, verbose=False):

    if random.random() < random_thres:
        brightness_factor = random.uniform(0.1, 1.2)
        img = TF.adjust_brightness(img, brightness_factor)
        if verbose:
            print('Brightness:', brightness_factor)

    if random.random() < random_thres:
        contrast_factor = random.uniform(0.2, 1.8)
        img = TF.adjust_contrast(img, contrast_factor)
        if verbose:
            print('Contrast:', contrast_factor)

    if random.random() < random_thres:
        # hue_factor = random.uniform(-0.1, 0.1)
        hue_factor = 0.1
        img = TF.adjust_hue(img, hue_factor)
        if verbose:
            print('Hue:', hue_factor)

    return img

def random_affine_transformation(img, mask, label, verbose=False):
    
    if random.random() < random_thres:
        degrees = random.uniform(-20, 20)
        translate_h = random.uniform(-0.2, 0.2)
        translate_v = random.uniform(-0.2, 0.2)
        scale = random.uniform(0.7, 1.3)
        shear = random.uniform(-20, 20)
        resample = TF.InterpolationMode.BICUBIC

        img = TF.affine(img, degrees, (translate_h, translate_v), scale, shear, resample)
        if mask:
            mask = TF.affine(mask, degrees, (translate_h, translate_v), scale, shear, resample)
        label = TF.affine(label, degrees, (translate_h, translate_v), scale, shear, resample)

        if verbose:
            print('Affine degrees: %.1f, T_h: %.1f, T_v: %.1f, Scale: %.1f, Shear: %.1f' % \
                (degrees, translate_h, translate_v, scale, shear))

    if random.random() < 0.5:
        
        img = TF.hflip(img)
        if mask:
            mask = TF.hflip(mask)
        label = TF.hflip(label)

        if verbose:
            print('Horizontal flip')

    if mask:
        return img, mask, label
    else:
        return img, label

def random_mask_perturbation(mask, verbose=False):

    degrees = random.uniform(-10, 10)
    translate_h = random.uniform(-0.1, 0.1)
    translate_v = random.uniform(-0.1, 0.1)
    scale = random.uniform(0.8, 1.2)
    shear = random.uniform(-10, 10)
    resample = TF.InterpolationMode.BICUBIC

    mask = TF.affine(mask, degrees, (translate_h, translate_v), scale, shear, resample)

    if verbose:
        print('Mask pertubation degrees: %.1f, T_h: %.1f, T_v: %.1f, Scale: %.1f, Shear: %.1f' % \
            (degrees, translate_h, translate_v, scale, shear))

    morphologic_times = int(random.random() * 10)
    morphologic_thres = random.random()
    filter_size = 7
    for i in range(morphologic_times):
        if random.random() < morphologic_thres:
            mask = mask.filter(ImageFilter.MinFilter(filter_size))
            if verbose:
                print(i, 'erossion')
        else:
            mask = mask.filter(ImageFilter.MaxFilter(filter_size))
            if verbose:
                print(i, 'dilation')

    mask = mask.convert('1')

    return mask

def random_resized_crop(img, mask, label, size, verbose=False):

    scale = (0.08, 1.0)
    ratio = (0.75, 1.33333333)

    sample_flag = False

    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            y = random.randint(0, img.size[1] - h)
            x = random.randint(0, img.size[0] - w)
            sample_flag = True
            break

    # Fallback
    if not sample_flag:
        w = min(img.size[0], img.size[1])
        y = (img.size[1] - w) // 2
        x = (img.size[0] - w) // 2
        h = w

    img = TF.resized_crop(img, y, x, h, w, size, TF.InterpolationMode.BICUBIC)
    if mask:
        mask = TF.resized_crop(mask, y, x, h, w, size, TF.InterpolationMode.BICUBIC)
    label = TF.resized_crop(label, y, x, h, w, size, TF.InterpolationMode.BICUBIC)

    if verbose:
        print('x: %d, y: %d, w: %d, h: %d' % (x, y, w, h))

    if mask:
        return img, mask, label
    else:
        return img, label

def imagenet_normalization(img_tensor):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tensor = TF.normalize(img_tensor, mean, std)

    return img_tensor
