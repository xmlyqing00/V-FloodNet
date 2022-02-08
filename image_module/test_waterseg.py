"""
Brian Tsai

Additional required libraries:
segmentation-models-pytorch
https://github.com/qubvel/segmentation_models.pytorch

To Execute:
python test_waterseg.py [--model_path PATH] [--test_path PATH] [--palette_path PATH OPTIONAL] [--out_path PATH OPTIONAL]

Example:

Folder test:
python test_waterseg.py --model_path "D:\DATASETS\models\model.pth" --test_path "D:\DATASETS\TESTDATA\datafolder"

Individual test:
python test_waterseg.py --model_path "D:\DATASETS\models\model.pth" --test_path "D:\DATASETS\TESTDATA\test_file.jpg"
"""
import sys
import os
import argparse
import time

import numpy as np

from PIL import Image
from pathlib import Path
from glob import glob

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF

ROOT_DIR = str(Path(__file__).resolve().parents[1])

time_str = timestr = time.strftime("%Y-%m-%d %H-%M-%S")
DEFAULT_OUT = os.path.join(ROOT_DIR, 'output', 'test_waterseg', time_str)
DEFAULT_PALETTE = os.path.join(ROOT_DIR, 'video_module', "assets", "mask_palette.png")
sys.path.append(ROOT_DIR)
print("Added", ROOT_DIR, "to PATH.")

def norm_imagenet(img_pil, dims):
    """
    Normalizes and resizes input image
    :param img_pil: PIL Image
    :param dims: Model's expected input dimensions
    :return: Normalized Image as a Tensor
    """

    # Mean and stddev of ImageNet dataset
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Resize, convert to tensor, normalize
    transform_norm = tf.Compose([
        tf.Resize([dims[0], dims[1]]),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])

    img_norm = transform_norm(img_pil)
    return img_norm


def predict_one(path, model, palette_path, mask_outdir, overlay_outdir):
    """
    Predicts a single image from path
    :param path: Path to image
    :param model: Loaded Torch Model
    :param palette_path: Path to palette file for overlay and mask output
    :param mask_outdir: Filepath to mask out directory
    :param overlay_outdir: Filepath to overlay out directory
    :return: None
    """
    img_pil = Image.open(path)

    # Prediction is an PIL Image of 0s and 1s
    prediction = predict_pil(model, img_pil, palette_path, model_dims=(416, 416))

    basename = str(Path(os.path.basename(path)).stem)
    mask_savepth = os.path.join(mask_outdir, basename + '.png')
    mask_save = prediction.convert('RGB')
    mask_save.save(mask_savepth)

    over_savepth = os.path.join(overlay_outdir, basename + '.png')
    overlay_np = np.array(img_pil) * 1 + np.array(mask_save) * 0.8
    overlay_np = overlay_np.clip(0, 255)
    Image.fromarray(overlay_np.astype(np.uint8)).save(over_savepth)


def predict_pil(model, img_pil, palette_path, model_dims):
    """
    Predicts a single PIL Image
    :param model: Loaded PyTorch model
    :param img_pil: PIL image
    :param palette_path: Palette filepath for reconstruction
    :param model_dims: Model input dimensions
    :return: Segmentation prediction as PIL Image
    """
    palette = Image.open(palette_path).getpalette()

    img_np = np.array(img_pil)
    img_tensor_norm = norm_imagenet(img_pil, model_dims)

    # Pipeline to resize the prediction to the original image dimensions
    pred_resize = tf.Compose([tf.Resize([img_np.shape[0], img_np.shape[1]])])

    # Add extra dimension at front as model expects input 1*3*dimX*dimY (batch size of 1)
    input_data = img_tensor_norm.unsqueeze(0)

    try:
        print("Converted input image to cuda.")
        prediction = model.predict(input_data.cuda())
    except:
        print("Did not convert input image to cuda.")
        prediction = model.predict(input_data)

    prediction = pred_resize(prediction)
    prediction = TF.to_pil_image(prediction.squeeze().cpu().round().int()).convert('P')
    prediction.putpalette(palette)
    return prediction


def test_waterseg(args):
    """
    Tests either a single or an entire folder of images
    :param args: Command line args
    :return: None
    """
    model = torch.load(args.model_path)
    test_path = args.test_path
    out_path = args.out_path

    mask_out = os.path.join(out_path, 'masks')
    overlay_out = os.path.join(out_path, 'overlay')
    if not os.path.exists(mask_out):
        os.makedirs(mask_out)
    if not os.path.exists(overlay_out):
        os.makedirs(overlay_out)

    if os.path.isfile(test_path):
        predict_one(test_path, model, args.palette_path, mask_out, overlay_out)
    elif os.path.isdir(test_path):
        paths = glob(os.path.join(test_path, '*.jpg')) + glob(os.path.join(test_path, '*.png'))
        for path in paths:
            predict_one(path, model, args.palette_path, mask_out, overlay_out)
    else:
        print("Error: Unknown path type:", test_path)


if __name__ == '__main__':
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch WaterNet Model Testing')
    # Required: Path to the .pth file.
    parser.add_argument('--model_path',
                        type=str,
                        metavar='PATH',
                        help='Path to the model')
    # Required: Path to either the single file or directory of files containing .jpg or .png images
    parser.add_argument('--test_path',
                        type=str,
                        metavar='PATH',
                        help='Can point to folder or an individual jpg/png image')
    # Optional: Defaults to the palette file in video_module/assets
    parser.add_argument('--palette_path',
                        default=DEFAULT_PALETTE,
                        type=str,
                        metavar='PATH',
                        help='(OPTIONAL) Path to palette file, defaults to file in video_module/assets')
    # Optional: Defaults to the output file in the project root/test_waterseg/<Date and Time at Runtime>
    # Produces two folders: 'masks' to contain the raw palette-based masks
    #                       'overlay' to contain an overlay
    parser.add_argument('--out_path',
                        default=DEFAULT_OUT,
                        type=str,
                        metavar='PATH',
                        help='(OPTIONAL) Path to output folder, defaults to project root/output')
    _args = parser.parse_args()

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    test_waterseg(_args)
    print("Done.")
