import numpy as np
from tqdm import tqdm, trange
import os
import argparse
from PIL import Image
from glob import glob
import torch
from torch import utils
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import subprocess

from video_module.dataset import Video_DS
from video_module.model import AFB_URR, FeatureBank
from test_image_seg import test_waterseg
import myutils

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description='Eval AFB-URR')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--budget', type=int, default='250000',
                        help='Max number of features that feature bank can store. Default: 300000')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--update-rate', type=float, default=0.1,
                        help='Update Rate. Impact of merging new features.')
    parser.add_argument('--merge-thres', type=float, default=0.95,
                        help='Merging Rate. If similarity higher than this, then merge, else append.')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Video Path')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Video Name')
    return parser.parse_args()


def main(args, device):
    model = AFB_URR(device, update_bank=True, load_imagenet_params=False)
    model = model.to(device)
    model.eval()

    downsample_size = 480

    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path)
        end_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'], strict=False)
        train_loss = checkpoint['loss']
        seed = checkpoint['seed']
        print(myutils.gct(),
              f'Loaded checkpoint {args.model_path}. (end_epoch: {end_epoch}, train_loss: {train_loss}, seed: {seed})')
    else:
        print(myutils.gct(), f'No checkpoint found at {args.model_path}')
        raise IOError

    img_list = sorted(glob(os.path.join(args.test_path, '*.jpg')) + glob(os.path.join(args.test_path, '*.png')))
    first_frame = myutils.load_image_in_PIL(img_list[0])
    first_name = os.path.basename(img_list[0])[:-4]

    image_model_out_dir = './output/test_image_seg'
    mask_dir = os.path.join(image_model_out_dir, args.test_name, 'mask')
    mask_path = os.path.join(mask_dir, first_name + '.png')
    if not os.path.exists(mask_path):
        image_model_path = './records/link_efficientb4_model.pth'
        test_waterseg(image_model_path, img_list[0], args.test_name, image_model_out_dir, device)

    first_mask = myutils.load_image_in_PIL(mask_path, 'P')
    seq_dataset = Video_DS(img_list, first_frame, first_mask)

    seq_loader = utils.data.DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=1)
    out_dir = './output/test_video_seg'
    seg_dir = os.path.join(out_dir, args.test_name, 'mask')
    os.makedirs(seg_dir, exist_ok=True)
    if args.viz:
        overlay_dir = os.path.join(out_dir, args.test_name, 'overlay')
        os.makedirs(overlay_dir, exist_ok=True)

    obj_n = seq_dataset.obj_n
    fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate, thres_close=args.merge_thres)

    ori_first_frame = seq_dataset.first_frame.unsqueeze(0).to(device)
    ori_first_mask = seq_dataset.first_mask.unsqueeze(0).to(device)

    first_frame = TF.resize(ori_first_frame, downsample_size, InterpolationMode.BICUBIC)
    first_mask = TF.resize(ori_first_mask, downsample_size, InterpolationMode.NEAREST)

    pred = torch.argmax(ori_first_mask[0], dim=0).cpu().numpy().astype(np.uint8)
    seg_path = os.path.join(seg_dir, f'{first_name}.png')
    myutils.save_seg_mask(pred, seg_path, palette)

    if args.viz:
        overlay_path = os.path.join(overlay_dir, f'{first_name}.png')
        myutils.save_overlay(ori_first_frame[0], pred, overlay_path, palette)

    with torch.no_grad():
        k4_list, v4_list = model.memorize(first_frame, first_mask)
        fb.init_bank(k4_list, v4_list)

        for idx, (frame, frame_name) in enumerate(tqdm(seq_loader)):

            ori_frame = frame.to(device)
            ori_size = ori_frame.shape[-2:]
            frame = TF.resize(ori_frame, downsample_size, InterpolationMode.BICUBIC)
            score, _ = model.segment(frame, fb)
            pred_mask = F.softmax(score, dim=1)

            k4_list, v4_list = model.memorize(frame, pred_mask)
            fb.update(k4_list, v4_list, idx + 1)

            pred = TF.resize(pred_mask, ori_size, InterpolationMode.BICUBIC)
            pred = torch.argmax(pred[0], dim=0).cpu().numpy().astype(np.uint8)
            pred = myutils.postprocessing_pred(pred)
            seg_path = os.path.join(seg_dir, f'{frame_name[0]}.png')
            myutils.save_seg_mask(pred, seg_path, palette)
            if args.viz:
                overlay_path = os.path.join(overlay_dir, f'{frame_name[0]}.png')
                myutils.save_overlay(ori_frame[0], pred, overlay_path, palette)

    fb.print_peak_mem()


if __name__ == '__main__':

    args = get_args()
    print(myutils.gct(), 'Args =', args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    assert os.path.isdir(args.test_path)

    palette = [0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 0]

    main(args, device)

    print(myutils.gct(), 'Test video segmentation done.')
