import numpy as np
from tqdm import tqdm, trange
import os
import argparse
from PIL import Image
import json
import torch
from torch import utils
from torch.nn import functional as F

from dataset import Video_DS
from model import AFB_URR, FeatureBank
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
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--prefix', type=str, default='fulltrain',
                        help='Prefix to the model name.')
    parser.add_argument('--update-rate', type=float, default=0.1,
                        help='Update Rate. Impact of merging new features.')
    parser.add_argument('--merge-thres', type=float, default=0.95,
                        help='Merging Rate. If similarity higher than this, then merge, else append.')
    parser.add_argument('--video', type=str, required=True,
                        help='Video Name')
    parser.add_argument('--label_num', type=int, default=1,
                        help='Path to the annotation folder.')
    parser.add_argument('--config', type=str, default='eval_config.json',
                        help='Path to eval config.')
    return parser.parse_args()


def eval_LongVideo(model, model_name, seq_name, seq_dataset):

    seq_loader = utils.data.DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=1)

    seg_dir = os.path.join('../output', model_name, seq_name)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    if args.viz:
        overlay_dir = os.path.join('./overlay', model_name, seq_name)
        if not os.path.exists(overlay_dir):
            os.makedirs(overlay_dir)

    masks_p = 0
    obj_n = seq_dataset.obj_n
    fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate, thres_close=args.merge_thres)

    # first_frame = seq_dataset.first_frame.unsqueeze(0).to(device)
    # first_mask = seq_dataset.masks[0].unsqueeze(0).to(device)
    # frame_n = seq_dataset.video_len

    # pred_mask = first_mask
    # pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
    # seg_path = os.path.join(seg_dir, f'{seq_dataset.first_name}.png')
    # myutils.save_seg_mask(pred, seg_path, palette)
    #
    # if args.viz:
    #     overlay_path = os.path.join(overlay_dir, f'{seq_dataset.first_name}.png')
    #     myutils.save_overlay(first_frame[0], pred, overlay_path, palette)

    # memorize
    gt_list = []
    gt_insert_list = []
    gt_insert_idx = 0
    for i in range(seq_dataset.mask_len):
        frame_idx = seq_dataset.masks_idx[i]
        frame, frame_name = seq_dataset[frame_idx]
        frame = frame.unsqueeze(0).to(device)
        gt_mask = seq_dataset.masks[i].unsqueeze(0).to(device)
        with torch.no_grad():
            k4_list, v4_list = model.memorize(frame, gt_mask)

        gt_list.append((k4_list, v4_list))
        if i == 0:
            gt_insert_list.append(0)
        else:
            gt_insert_list.append((seq_dataset.masks_idx[i] + seq_dataset.masks_idx[i - 1]) // 2)


    for idx, (frame, frame_name) in enumerate(tqdm(seq_loader, desc=f'{seq_name}')):

        frame = frame.to(device)

        # Activate gt
        if gt_insert_idx < seq_dataset.mask_len and idx == gt_insert_list[gt_insert_idx]:
            fb.append(gt_list[gt_insert_idx][0], gt_list[gt_insert_idx][1], idx)
            gt_insert_idx += 1

        if masks_p < seq_dataset.mask_len and idx == seq_dataset.masks_idx[masks_p]:
            gt_mask = seq_dataset.masks[masks_p].unsqueeze(0).to(device)
            pred_mask = gt_mask
            masks_p += 1
            gt_flag = True
        else:
            score, _ = model.segment(frame, fb)
            pred_mask = F.softmax(score, dim=1)
            gt_flag = False

        with torch.no_grad():
            k4_list, v4_list = model.memorize(frame, pred_mask)
        pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, f'{frame_name[0]}.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if gt_flag:
            fb.update(k4_list, v4_list, idx + 1, update_rate=1)
        else:
            fb.update(k4_list, v4_list, idx + 1)

        if args.viz:
            overlay_path = os.path.join(overlay_dir, f'{frame_name[0]}.png')
            myutils.save_overlay(frame[0], pred, overlay_path, palette)

    fb.print_peak_mem()


def main():
    model = AFB_URR(device, update_bank=True, load_imagenet_params=False)
    model = model.to(device)
    model.eval()

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            end_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'], strict=False)
            train_loss = checkpoint['loss']
            seed = checkpoint['seed']
            print(myutils.gct(),
                  f'Loaded checkpoint {args.resume}. (end_epoch: {end_epoch}, train_loss: {train_loss}, seed: {seed})')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError


    model_name = 'AFB-URR_Water'
    with open(args.config, 'r') as f:
        eval_config = json.load(f)

    if not eval_config.get(args.video):
        raise ValueError(f'Please update the video information {args.video} in the config file.')

    img_dir = os.path.join(eval_config['img_dir'], args.video)
    label_dir = os.path.join(eval_config['label_dir'], args.video)
    label_list = eval_config[args.video]['labels'][:args.label_num]

    seq_dataset = Video_DS(img_dir, label_dir, label_list)
    seq_name = f'{args.video}_label_{args.label_num}'

    if args.prefix:
        model_name += f'_{args.prefix}'

    print(myutils.gct(), f'Model name: {model_name}')

    eval_LongVideo(model, model_name, seq_name, seq_dataset)


if __name__ == '__main__':

    args = get_args()
    print(myutils.gct(), 'Args =', args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    palette = Image.open(os.path.join('./assets/mask_palette.png')).getpalette()

    main()

    print(myutils.gct(), 'Evaluation done.')
