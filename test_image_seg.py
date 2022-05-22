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
import os
import argparse
import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from glob import glob
from tqdm import tqdm

import torch
import torchvision.transforms as tf

import myutils

# ROOT_DIR = str(Path(__file__).resolve().parents[0])
ROOT_DIR = './'
# time_str = timestr = time.strftime("%Y-%m-%d %H-%M-%S")
# DEFAULT_OUT = os.path.join(ROOT_DIR, 'output', 'test_waterseg', time_str)
DEFAULT_OUT = os.path.join(ROOT_DIR, 'output', 'segs')
# DEFAULT_PALETTE = os.path.join(ROOT_DIR, "assets", "mask_palette.png")
# sys.path.append(ROOT_DIR)
# print("Added", ROOT_DIR, "to PATH.")


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


def predict_one(path, model, mask_outdir, overlay_outdir, device):
    """
    Predicts a single image from path
    :param path: Path to image
    :param model: Loaded Torch Model
    :param mask_outdir: Filepath to mask out directory
    :param overlay_outdir: Filepath to overlay out directory
    :return: None
    """
    img_pil = myutils.load_image_in_PIL(path)

    # Prediction is an PIL Image of 0s and 1s
    prediction = predict_pil(model, img_pil, model_dims=(416, 416), device=device)

    basename = str(Path(os.path.basename(path)).stem)
    mask_savepth = os.path.join(mask_outdir, basename + '.png')
    # mask_save = prediction.convert('RGB')
    prediction.save(mask_savepth)

    over_savepth = os.path.join(overlay_outdir, basename + '.png')
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay_np = myutils.add_overlay(img_np, np.array(prediction))
    cv2.imwrite(over_savepth, overlay_np)
    # overlay_np = np.array(img_pil) * 1 + np.array(prediction.convert('RGB')) * 0.8
    # overlay_np = overlay_np.clip(0, 255)
    # Image.fromarray(overlay_np.astype(np.uint8)).save(over_savepth)


def predict_pil(model, img_pil, model_dims, device):
    """
    Predicts a single PIL Image
    :param model: Loaded PyTorch model
    :param img_pil: PIL image
    :param model_dims: Model input dimensions
    :return: Segmentation prediction as PIL Image
    """

    img_np = np.array(img_pil)
    img_tensor_norm = norm_imagenet(img_pil, model_dims)

    # Pipeline to resize the prediction to the original image dimensions
    pred_resize = tf.Compose([tf.Resize([img_np.shape[0], img_np.shape[1]])])

    # Add extra dimension at front as model expects input 1*3*dimX*dimY (batch size of 1)
    input_data = img_tensor_norm.unsqueeze(0)

    try:
        # print("Converted input image to cuda.")
        prediction = model.predict(input_data.to(device))
    except:
        print("Did not convert input image to cuda.")
        prediction = model.predict(input_data)

    prediction = pred_resize(prediction)
    prediction = myutils.postprocessing_pred(prediction.squeeze().cpu().round().numpy().astype(np.uint8))
    prediction = Image.fromarray(prediction).convert('P')
    prediction.putpalette(myutils.color_palette)
    return prediction


def test_waterseg(model_path, test_path, test_name, out_path, device):
    """
    Tests either a single or an entire folder of images
    :param args: Command line args
    :return: None
    """
    model = torch.load(model_path)
    test_path = test_path
    out_path = os.path.join(out_path, test_name)

    mask_out = os.path.join(out_path, 'mask')
    overlay_out = os.path.join(out_path, 'overlay')
    if not os.path.exists(mask_out):
        os.makedirs(mask_out)
    if not os.path.exists(overlay_out):
        os.makedirs(overlay_out)

    if os.path.isfile(test_path):
        predict_one(test_path, model, mask_out, overlay_out, device)
    elif os.path.isdir(test_path):
        paths = glob(os.path.join(test_path, '*.jpg')) + glob(os.path.join(test_path, '*.png'))
        for path in tqdm(paths):
            predict_one(path, model, mask_out, overlay_out, device)
    else:
        print("Error: Unknown path type:", test_path)


if __name__ == '__main__':
    # Hyper parameters
    parser = argparse.ArgumentParser(description='V-FloodNet Video WaterNet Model Testing')
    # Required: Path to the .pth file.
    parser.add_argument('--model_path',
                        default='./records/link_efficientb4_model.pth',
                        type=str,
                        metavar='PATH',
                        help='Path to the model')
    # Required: Path to either the single file or directory of files containing .jpg or .png images
    parser.add_argument('--test_path',
                        type=str,
                        metavar='PATH',
                        required=True,
                        help='Can point to folder or an individual jpg/png image')
    parser.add_argument('--test_name',
                        type=str,
                        required=True,
                        help='Test name')
    parser.add_argument('--out_path',
                        default=DEFAULT_OUT,
                        type=str,
                        metavar='PATH',
                        help='(OPTIONAL) Path to output folder, defaults to project root/output')
    args = parser.parse_args()

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    test_waterseg(args.model_path, args.test_path, args.test_name, args.out_path, device)

    print(myutils.gct(), 'Test image segmentation done.')
