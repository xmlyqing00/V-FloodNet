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


time_fmt = mdates.DateFormatter('%m-%d %H:%M')
# register_matplotlib_converters()
fontsize = 24
rotation = 45

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


def get_parser():
    parser = argparse.ArgumentParser(description='Compare water level.')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of the test video')
    # parser.add_argument('--water_mask_dir', type=str, default='./output', required=True,
    #                     help='Path to the water mask folder.')
    # parser.add_argument('--img_dir', type=str, required=True,
    #                     help='Input image directory.')
    parser.add_argument('--out-dir', default='output/waterlevel',
                        help='A file or directory to save output results.')

    # parser.add_argument('--video', type=str, default='boston_harbor_20190119_20190123_day_s', help='Video name.')
    # parser.add_argument('--recalib', action='store_true', help='Recalibate the video')
    # parser.add_argument('--reref', action='store_true', help='Re-pick the reference objects in the video')
    # parser.add_argument('--plot', action='store_true', help='Recalibate the video')
    # parser.add_argument('--fit', action='store_true', help='Fit the px curve to the meter curve.')
    # parser.add_argument('--ref_num', type=int, default=3, help='Reference object num.')
    args = parser.parse_args()
    return args


def main(args):

    out_dir = os.path.join(args.out_dir, f'{args.test_name}-{args.opt}')
    gt_dir = './records/groundtruth'

    print('Load waterlevel_px.npy and timestamp_list.npy')
    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.npy')
    waterlevel_px = np.load(waterlevel_path)

    timestamp_list_path = os.path.join(out_dir, 'timestamp_list.npy')
    timestamp_list = np.load(timestamp_list_path, allow_pickle=True)

    gt_path = os.path.join(gt_dir, f'{args.test_name}_gt.csv')
    gt_csv = pd.read_csv(gt_path)

    px_to_meter_path = os.path.join(gt_dir, f'{args.test_name}_px_to_meter.txt')
    px_to_meter = np.loadtxt(px_to_meter_path)

    if 'boston_harbor' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1])
        timestamp_list_gt = timestamp_list_gt - timedelta(minutes=60)
        gt_col_id = 4
        # tick_spacing = 6
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        # ticker_locator = mdates.HourLocator(interval=tick_spacing)
    elif 'houston' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%m/%d/%Y %H:%M')
        gt_col_id = 2
        # tick_spacing = 6
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        # ticker_locator = mdates.HourLocator(interval=tick_spacing)
    elif 'LSU' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], errors='coerce', format='%Y-%m-%d-%H-%M-%S')
        gt_col_id = 1
        # tick_spacing = len(time_arr_gt) // 10
        # ticker_locator = ticker.MultipleLocator(tick_spacing)
        # ticker_locator = mdates.HourLocator(interval=tick_spacing)
        # ticker_locator = mdates.MinuteLocator(interval=tick_spacing)
    else:
        raise NotImplementedError

    waterlevel_meter = px_to_meter[0] * waterlevel_px + px_to_meter[1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], '-', linewidth=3, label=f'Groundtruth')
    ax.plot(timestamp_list_gt, gt_csv.iloc[:, gt_col_id], '^', markersize=15, label=f'Groundtruth')

    # for i in range(waterlevel_px.shape[0]):
    #     ax.plot(time_arr_eval, waterlevel_px[i, :], '.', label=f'By ref {i} (ft)')

    if 'houston' in args.test_name:
        # ax.plot(time_arr_eval, waterlevel_meter[0], '-', linewidth=3, label=f'Est Water Level0 (m)')
        # ax.plot(time_arr_eval, waterlevel_meter[1], '-', linewidth=3, label=f'Est Water Level1 (m)')
        ax.plot(timestamp_list, waterlevel_meter, '-', linewidth=3, label=f'Est Water Level')
        old_col_id = 5
        ax.plot(timestamp_list, gt_csv.iloc[:, old_col_id], '-', linewidth=3, label=f'LSUSeg Water Level')
        ax.axhline(y=10.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=fontsize)
    else:
        ax.plot(timestamp_list, waterlevel_meter, 'o', markersize=15, label=f'Est Water Level')
        ax.legend(loc='lower right', fontsize=fontsize)

    ticker_locator = mdates.AutoDateLocator()
    ticker_locator.intervald[mdates.HOURLY] = [4]
    ticker_locator.intervald[mdates.MINUTELY] = [3]
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

    _args = get_parser()
    _args.opt = 'ref'
    print(_args)

    main(_args)
