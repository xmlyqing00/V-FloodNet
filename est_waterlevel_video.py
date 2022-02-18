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

register_matplotlib_converters()

rectify_window_name = 'Select 4 pts to get homography matrix'
pts = []
pts_n = 4
loop_flag = True


def mouse_click(event, x, y, flags, param):
    global pts, loop_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv2.EVENT_LBUTTONUP:

        cv2.circle(param, pts[-1], 5, (0, 0, 200), -1)
        cv2.imshow(rectify_window_name, param)

        if len(pts) == pts_n:
            loop_flag = False


def get_video_homo(img_dir, homo_mat_path, recalib):
    # Order: left top, right top, left bottom, right bottom
    if not recalib and os.path.exists(homo_mat_path):
        print(f'Load homo mat from {homo_mat_path}')
        homo_mat = np.asmatrix(np.loadtxt(homo_mat_path))
        return homo_mat

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda x: (len(x), x))
    img_st = cv2.imread(os.path.join(img_dir, img_list[0]))

    print('Estimate the video homo mat.')

    global pts
    cv2.namedWindow(rectify_window_name)
    cv2.setMouseCallback(rectify_window_name, mouse_click, param=img_st.copy())

    cv2.imshow(rectify_window_name, img_st)
    while loop_flag:
        cv2.waitKey(1)

    print('Point src:', pts)
    d_x = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
    d_y = ((pts[2][0] - pts[0][0]) ** 2 + (pts[2][1] - pts[0][1]) ** 2) ** 0.5
    pts_t = [pts[0],
             (pts[0][0] + d_x, pts[0][1]),
             (pts[0][0], pts[0][1] + d_y),
             (pts[0][0] + d_x, pts[0][1] + d_y)]
    print('Point dst (Top Left, Top Right, Bottom Left, Bottom Right):', pts_t)

    pts = np.float32(pts)
    pts_t = np.float32(pts_t)
    homo_mat, _ = cv2.findHomography(pts, pts_t)

    np.savetxt(homo_mat_path, homo_mat)
    cv2.destroyWindow(rectify_window_name)

    return homo_mat


def get_seg_results(homo_mat, img_list, seg_list, l, r):
    img_data_list = []
    seg_data_list = []
    overlay_data_list = []
    time_list = []
    name_list = []
    for i in trange(l, r, desc='Load'):
        img = cv2.imread(img_list[i])
        img_size = (img.shape[1], img.shape[0])
        img = cv2.warpPerspective(img, homo_mat, img_size)

        seg = np.array(myutils.load_image_in_PIL(seg_list[i], 'P'))
        seg = cv2.warpPerspective(seg, homo_mat, img_size)
        overlay = myutils.add_overlay(img, seg, mask_palette)

        name = os.path.basename(img_list[i])[:-4]
        timestamp = datetime.strptime(name, '%Y-%m-%d-%H-%M-%S')

        img_data_list.append(img)
        seg_data_list.append(seg)
        overlay_data_list.append(overlay)
        time_list.append(timestamp)
        name_list.append(name)

    return img_data_list, seg_data_list, overlay_data_list, time_list, name_list


def get_waterlevel_multi_objs(seg_data_list, overlay_data_list):
    n = len(seg_data_list)
    gauge_label_id = 2
    waterlevel_px = np.zeros((2, n))

    for i in trange(n):

        gauge_pos = (seg_data_list[i] == gauge_label_id).nonzero(as_tuple=False)
        gauge_top_pos = (int(gauge_pos[1].mean()), int(gauge_pos[0].min()) + 3)

        if i > 0:
            waterlevel_px[:, i] = waterlevel_px[:, i - 1]

        gauge_end_flag = False

        for y in range(gauge_top_pos[1] + 5, seg_data_list[i].shape[0]):

            if not gauge_end_flag and seg_data_list[i][y][gauge_top_pos[0]] != gauge_label_id:
                waterlevel_px[0, i] = y - gauge_top_pos[1]
                gauge_end_flag = True

            if seg_data_list[i][y][gauge_top_pos[0]] == water_label_id:
                waterlevel_px[1, i] = y - gauge_top_pos[1]
                cv2.line(overlay_data_list[i], gauge_top_pos, (gauge_top_pos[0], y), (200, 0, 0), 2)
                break

    waterlevel_px = waterlevel_px[:, 0:1] - waterlevel_px

    return waterlevel_px


def get_waterlevel_user_selection(img_data_list, seg_data_list, overlay_data_list, reref=False):
    ref_bbox_path = os.path.join(info_dir, 'ref_bbox.txt')
    ref_img_path = os.path.join(info_dir, 'ref_img.jpg')
    if not reref and os.path.exists(ref_bbox_path):
        print('Load bbox of the reference objects.', ref_bbox_path)
        ref_bbox = np.loadtxt(ref_bbox_path)
        if type(ref_bbox[0]) == np.float64:
            ref_bbox = ref_bbox[np.newaxis, :]
        if enable_tracker:
            print('Load the reference image.', ref_img_path)
            ref_img = cv2.imread(ref_img_path)
        else:
            ref_img = img_data_list[0]
        ref_num = ref_bbox.shape[0]
        cv2.imwrite(ref_img_path, ref_img)
    else:
        ref_num = args.ref_num
        ref_img = img_data_list[0]
        cv2.imwrite(ref_img_path, ref_img)
        ref_bbox = []

    n = len(img_data_list)
    waterlevel_px = np.zeros((ref_num, n))
    ref_bbox_final = []

    for ref_idx in range(ref_num):

        # Get points of reference objs
        if reref or ref_idx == len(ref_bbox):

            track_window_name = 'Select A Rect As Reference Obj'
            while True:
                bbox_st = cv2.selectROI(track_window_name, ref_img, fromCenter=False)
                if bbox_st[2] > 0 and bbox_st[3] > 0:
                    break
            cv2.destroyWindow(track_window_name)
            ref_bbox.append(bbox_st)
        else:
            bbox_st = tuple(ref_bbox[ref_idx])

        # x, y, w, h = [int(v) for v in bbox_st]
        # cv2.rectangle(overlay_data_list[0], (x, y), (x + w, y + h), (0, 200, 0), 2)

        if enable_tracker:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(ref_img, bbox_st)
        bbox_last = bbox_st
        key_pts = []

        for i in trange(n, desc=f'Est by Ref {ref_idx}'):

            if enable_tracker:
                tracker_flag, bbox = tracker.update(img_data_list[i])
                if tracker_flag:
                    bbox_last = bbox
                else:
                    bbox = bbox_last
            else:
                bbox = bbox_last

            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(overlay_data_list[i], (x, y), (x + w, y + h), (0, 200, 0), 2)

            key_pt = (int(x + w / 2), int(y + h))
            key_pts.append(key_pt)

            if i > 0:
                waterlevel_px[ref_idx, i] = waterlevel_px[ref_idx, i - 1]

            for y in range(key_pt[1] + 1, seg_data_list[i].shape[0]):
                if seg_data_list[i][y][key_pts[i][0]] == water_label_id:
                    waterlevel_px[ref_idx, i] = y - key_pt[1]
                    cv2.line(overlay_data_list[i], key_pt, (key_pt[0], y), (200, 0, 0), 2)
                    break

        ref_bbox_final.append(bbox)

    ref_bbox_path = os.path.join(info_dir, 'ref_bbox.txt')
    np.savetxt(ref_bbox_path, np.array(ref_bbox))

    ref_bbox_path = os.path.join(info_dir, 'ref_bbox_final.txt')
    np.savetxt(ref_bbox_path, np.array(ref_bbox_final))

    return waterlevel_px


def est_waterlevel():
    # Calibrate the video
    homo_mat_path = os.path.join(info_dir, 'homo_mat.txt')
    homo_mat = get_video_homo(img_dir, homo_mat_path, args.recalib)

    # Load segmentation results
    img_list = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    img_list.sort(key=lambda x: (len(x), x))

    seg_list = glob(os.path.join(seg_dir, '*.png')) + glob(os.path.join(seg_dir, '*.jpg'))
    seg_list.sort(key=lambda x: (len(x), x))

    chunk_size = 500
    waterlevel_px = None
    time_list = []
    for i in range(0, len(img_list), chunk_size):
        chunk_right = min(len(img_list), i + chunk_size)
        print(f'Chunk range: {i} - {chunk_right}. Total {len(img_list)}')
        img_data_list, seg_data_list, overlay_data_list, time_list_chunk, name_list = \
            get_seg_results(homo_mat, img_list, seg_list, i, chunk_right)

        time_list += time_list_chunk
        if ref_obj_type == 0:
            waterlevel_px_chunk = get_waterlevel_multi_objs(seg_data_list, overlay_data_list)
        else:
            if i == 0:
                reref = args.reref
            else:
                reref = False
            waterlevel_px_chunk = get_waterlevel_user_selection(img_data_list, seg_data_list, overlay_data_list, reref)

        for i in trange(len(name_list), desc='Save Viz'):
            cv2.imwrite(os.path.join(out_dir, name_list[i] + '.png'), overlay_data_list[i])

        if waterlevel_px is None:
            waterlevel_px = waterlevel_px_chunk
        else:
            waterlevel_px = np.concatenate((waterlevel_px, waterlevel_px_chunk), axis=1)

    time_list_path = os.path.join(out_dir, 'time_list.npy')
    np.save(time_list_path, np.array(time_list))

    # Take negative values
    waterlevel_px = -waterlevel_px
    waterlevel_px = gaussian_filter1d(waterlevel_px, sigma=2, mode='nearest')

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.npy')
    np.save(waterlevel_path, waterlevel_px)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ref_num = waterlevel_px.shape[0]
    for ref_idx in range(ref_num):
        ax.plot(time_list, waterlevel_px[ref_idx], '-', label=f'Ref {ref_idx}')

    ax.legend(loc='lower right', fontsize=fontsize)
    ax.xaxis.set_major_formatter(time_fmt)
    ax.set_ylabel('Estimated Water Level (pixel)', fontsize=fontsize)
    # tick_spacing = 3
    # ticker_locator = mdates.MinuteLocator(tick_spacing)
    # ax.xaxis.set_major_locator(ticker_locator)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    ax.legend(loc='lower right', fontsize=fontsize)

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.png')
    fig.tight_layout()
    fig.savefig(waterlevel_path, dpi=300)


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

    parser = argparse.ArgumentParser(description='Estimate water level.')
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
