import os
import argparse
import configparser
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from PIL import Image
from datetime import datetime, timedelta
from glob import glob
import cv2
import argparse
import bisect
from pandas.plotting import register_matplotlib_converters
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq
from tqdm import tqdm, trange

import myutils

mask_palette_path = '/Ship01/Dataset/VOS/DAVIS-2017-train-val/mask_palette.png'
mask_palette = Image.open(mask_palette_path).getpalette()

water_label_id = 1

# time_fmt = mdates.DateFormatter('%m/%d %H:%M')
time_fmt = mdates.DateFormatter('%H:%M')


fontsize = 36
rotation = 90

def get_arr_gt_sample(arr_px, arr_gt, time_arr_px, time_arr_gt):
    ref_n, data_n = arr_px.shape
    arr_gt_sample = np.zeros(data_n)
    for i in range(len(time_arr_px)):
        k = bisect.bisect_left(time_arr_gt, time_arr_px[i])
        if k == 0:
            arr_gt_sample[i] = arr_gt[k]
        else:
            r = (time_arr_px[i] - time_arr_gt[k - 1]) / (time_arr_gt[k] - time_arr_gt[k - 1])
            arr_gt_sample[i] = arr_gt[k - 1] + r * (arr_gt[k] - arr_gt[k - 1])

    return arr_gt_sample


def fit(arr_px, arr_gt, time_arr_px, time_arr_gt):
    def fit_func(params, x):
        return params[0] * x + params[1]

    def fit_err(params, x, gt):
        y = fit_func(params, x)
        return y - gt

    arr_gt_sample = get_arr_gt_sample(arr_px, arr_gt, time_arr_px, time_arr_gt)
    ref_n = arr_px.shape[0]

    params_all = np.zeros((ref_n, 2))
    for i in range(ref_n):
        params_init = np.array([1, 0])
        params_fit, _ = leastsq(fit_err, params_init, args=(arr_px[i], arr_gt_sample))
        params_all[i] = params_fit

    return params_all


def fit_px_to_meter(out_dir, info_dir):
    print('Load waterlevel_px.npy and time_arr.npy')

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.npy')
    waterlevel_px = np.load(waterlevel_path)

    time_arr_path = os.path.join(out_dir, 'time_list.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)

    gt_path = os.path.join(info_dir, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)

    if 'boston_harbor' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1])
        time_arr_gt = time_arr_gt - timedelta(minutes=60)
        gt_col_id = 4

    elif 'houston' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%m/%d/%Y %H:%M')
        gt_col_id = 1

    elif 'LSU' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%Y-%m-%d-%H-%M')
        gt_col_id = 1

    else:
        raise NotImplementedError

    px_to_meter_path = os.path.join(info_dir, 'px_to_meter.txt')
    px_to_meter = fit(waterlevel_px, gt_csv.iloc[:, gt_col_id], time_arr_eval, time_arr_gt)
    np.savetxt(px_to_meter_path, px_to_meter)

    print(f'Save params to {px_to_meter_path}.')


def plot_hydrograph(out_dir, info_dir):
    print('Load waterlevel_px.npy and time_arr.npy')

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.npy')
    waterlevel_px = np.load(waterlevel_path)

    time_arr_path = os.path.join(out_dir, 'time_list.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)

    gt_path = os.path.join(info_dir, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)

    if 'boston_harbor' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1])
        time_arr_gt = time_arr_gt - timedelta(minutes=60)
        gt_col_id = 4
        tick_spacing = 6
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        ticker_locator = mdates.HourLocator(interval=tick_spacing)
    elif 'houston' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%m/%d/%Y %H:%M')
        gt_col_id = 2
        tick_spacing = 6
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        ticker_locator = mdates.HourLocator(interval=tick_spacing)
    elif 'LSU' in img_dir_name:
        time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0], errors='coerce', format='%Y-%m-%d-%H-%M-%S')
        gt_col_id = 1
        tick_spacing = len(time_arr_gt) // 10
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        # ticker_locator = mdates.HourLocator(interval=tick_spacing)
        ticker_locator = mdates.MinuteLocator(interval=tick_spacing)
    else:
        raise NotImplementedError

    ref_n = waterlevel_px.shape[0]
    px_to_meter_path = os.path.join(info_dir, 'px_to_meter.txt')
    if os.path.exists(px_to_meter_path):
        px_to_meter = np.loadtxt(px_to_meter_path)
        if len(px_to_meter.shape) == 1:
            px_to_meter = px_to_meter[np.newaxis, :]
    else:
        px_to_meter = np.array([[1, 0]] * ref_n)
        np.savetxt(px_to_meter_path, px_to_meter)

    waterlevel_px[waterlevel_px >= -1 - 1e-8] = np.NaN
    waterlevel_meter = px_to_meter[:, 0:1] * waterlevel_px + px_to_meter[:, 1:2]
    # for i in range(waterlevel_meter.shape[0]):
    #     plt.plot(waterlevel_meter[i])
    # plt.show()

    waterlevel_meter_avg = np.nanmean(waterlevel_meter, axis=0)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], '-', linewidth=3, label=f'Groundtruth')
    ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], '^', markersize=15, label=f'Groundtruth')

    # for i in range(waterlevel_px.shape[0]):
    #     ax.plot(time_arr_eval, waterlevel_px[i, :], '.', label=f'By ref {i} (ft)')

    if 'houston' in img_dir_name:
        # ax.plot(time_arr_eval, waterlevel_meter[0], '-', linewidth=3, label=f'Est Water Level0 (m)')
        # ax.plot(time_arr_eval, waterlevel_meter[1], '-', linewidth=3, label=f'Est Water Level1 (m)')
        ax.plot(time_arr_eval, waterlevel_meter_avg, '-', linewidth=3, label=f'Est Water Level')
        old_col_id = 5
        ax.plot(time_arr_eval, gt_csv.iloc[:, old_col_id], '-', linewidth=3, label=f'LSUSeg Water Level')
        ax.axhline(y=10.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=fontsize)
    else:
        ax.plot(time_arr_eval, waterlevel_meter_avg, 'o', markersize=15, label=f'Est Water Level')
        ax.legend(loc='lower right', fontsize=fontsize)

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    ax.set_ylabel('Water Level (meter)', fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    waterlevel_path = os.path.join(out_dir, 'waterlevel_meter.png')
    print(waterlevel_path)
    fig.tight_layout()
    fig.savefig(waterlevel_path, dpi=200)

    print(f'Save figure to {waterlevel_path}.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare water level.')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the test video')
    parser.add_argument('--water_mask_dir', type=str, default='./output', required=True,
                        help='Path to the water mask folder.')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Input image directory.')

    parser.add_argument('--video', type=str, default='boston_harbor_20190119_20190123_day_s', help='Video name.')
    parser.add_argument('--recalib', action='store_true', help='Recalibate the video')
    parser.add_argument('--reref', action='store_true', help='Re-pick the reference objects in the video')
    parser.add_argument('--plot', action='store_true', help='Recalibate the video')
    parser.add_argument('--fit', action='store_true', help='Fit the px curve to the meter curve.')
    parser.add_argument('--ref_num', type=int, default=3, help='Reference object num.')
    args = parser.parse_args()

    print('Args:', args)

    # Paths
    img_root = '/Ship01/Dataset/VOS/water'
    seg_root = 'output/AFB-URR_Water_fulltrain'
    out_root = 'output/waterlevel'

    img_dir_name = args.video[:args.video.index('_label_')]
    img_dir = os.path.join(img_root, 'JPEGImages', img_dir_name)
    info_dir = os.path.join(img_root, 'WaterlevelGT', img_dir_name)
    seg_dir = os.path.join(seg_root, args.video)
    out_dir = os.path.join(out_root, args.video)
    overlay_dir = os.path.join('overlay/AFB-URR_Water_fulltrain', args.video)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    ref_obj_type = 1
    # if 'LSU' in img_dir_name:
    #     ref_obj_type = 0
    # else:
    #     ref_obj_type = 1

    if 'boston_harbor' in img_dir_name or 'LSU' in img_dir_name:
        enable_tracker = True
    else:
        enable_tracker = False

    if args.fit:
        est_waterlevel()
        fit_px_to_meter(out_dir, info_dir)
        plot_hydrograph(out_dir, info_dir)
    else:
        if args.plot:
            plot_hydrograph(out_dir, info_dir)
        else:
            est_waterlevel()
            plot_hydrograph(out_dir, info_dir)
