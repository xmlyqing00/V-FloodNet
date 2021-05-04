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

time_fmt = mdates.DateFormatter('%m/%d %H:%M')
# print(time_arr)

tick_spacing = 6
# ticker_locator = ticker.MultipleLocator(tick_spacing)
ticker_locator = mdates.HourLocator(interval=tick_spacing)
font_size = 18

register_matplotlib_converters()

# Waterlevel
# params_delay = 5  # For boston_harbor_20190119_20190123_day

rectify_window_name = 'Select 4 pts to get homography matrix'
pts = []
pts_n = 4
loop_flag = True


def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def mouse_click(event, x, y, flags, param):
    global pts, loop_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv2.EVENT_LBUTTONUP:

        cv2.circle(param, pts[-1], 5, (0, 0, 200), -1)
        cv2.imshow(rectify_window_name, param)

        if len(pts) == pts_n:
            loop_flag = False


def get_video_homo(video_folder, homo_mat_path, recalib_flag):
    img_list = os.listdir(video_folder)
    img_list.sort(key=lambda x: (len(x), x))
    img_st = cv2.imread(os.path.join(video_folder, img_list[0]))

    if not recalib_flag and os.path.exists(homo_mat_path):
        print(f'Load homo mat from {homo_mat_path}')
        homo_mat = np.asmatrix(np.loadtxt(homo_mat_path))
        return homo_mat

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
    pts_t = [pts[0]]
    pts_t.append((pts_t[0][0] + d_x, pts_t[0][1]))
    pts_t.append((pts_t[0][0], pts_t[0][1] + d_y))
    pts_t.append((pts_t[0][0] + d_x, pts_t[0][1] + d_y))
    print('Point dst:', pts_t)

    pts = np.float32(pts)
    pts_t = np.float32(pts_t)
    homo_mat, _ = cv2.findHomography(pts, pts_t)

    np.savetxt(homo_mat_path, homo_mat)
    cv2.destroyWindow(rectify_window_name)

    return homo_mat


def track_object(img_dir, seg_dir, out_dir, homo_mat, sample_iter=0, bbox_st=None):
    tracker = cv2.TrackerCSRT_create()

    # img_list = os.listdir(img_dir)
    img_list = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    img_list.sort(key=lambda x: (len(x), x))

    seg_list = glob(os.path.join(seg_dir, '*.png')) + glob(os.path.join(seg_dir, '*.jpg'))
    seg_list.sort(key=lambda x: (len(x), x))

    # First frame
    frame_st = cv2.imread(img_list[0])
    img_size = (frame_st.shape[1], frame_st.shape[0])
    frame_st = cv2.warpPerspective(frame_st, homo_mat, img_size)

    if sample_iter == 0:
        seg_st = np.array(myutils.load_image_in_PIL(seg_list[0], 'P'))
        seg_st = cv2.warpPerspective(seg_st, homo_mat, img_size)
        overlay_st = myutils.add_overlay(frame_st, seg_st, mask_palette)
    else:
        overlay_st = cv2.imread(os.path.join(out_dir, os.path.basename(seg_list[0])))

    track_window_name = 'Select a RoI to track'
    # bbox_st = (768, 223, 46, 42)
    # Boston harbor bbox =  ((241, 38, 22, 10), (309, 52, 34, 22), (516, 77, 13, 35))
    if bbox_st is None:
        while True:
            bbox_st = cv2.selectROI(track_window_name, frame_st, fromCenter=False)
            if bbox_st[2] > 0 and bbox_st[3] > 0:
                break
        cv2.destroyWindow(track_window_name)

    x, y, w, h = [int(v) for v in bbox_st]
    cv2.rectangle(overlay_st, (x, y), (x + w, y + h), (0, 200, 0), 2)
    out_path = os.path.join(out_dir, os.path.basename(seg_list[0]))
    cv2.imwrite(out_path, overlay_st)

    print('Tracking box init position', bbox_st)

    tracker.init(frame_st, bbox_st)
    bbox_old = bbox_st

    key_pts = [(int(x + w / 2), int(y + h))]

    for i in trange(1, len(seg_list)):

        img = cv2.imread(img_list[i])
        img = cv2.warpPerspective(img, homo_mat, img_size)

        if sample_iter == 0:
            seg = np.array(myutils.load_image_in_PIL(seg_list[i], 'P'))
            seg = cv2.warpPerspective(seg, homo_mat, img_size)
            overlay = myutils.add_overlay(img, seg, mask_palette)
        else:
            overlay = cv2.imread(os.path.join(out_dir, os.path.basename(seg_list[i])))

        if enable_tracker:
            op_flag, bbox = tracker.update(img)
        else:
            op_flag, bbox = True, bbox_st

        # print(i, op_flag, bbox)
        if op_flag:
            x, y, w, h = [int(v) for v in bbox]
            bbox_old = bbox
        else:
            x, y, w, h = [int(v) for v in bbox_old]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 2)

        out_path = os.path.join(out_dir, os.path.basename(seg_list[i]))
        cv2.imwrite(out_path, overlay)

        # cv2.imshow('overlay', overlay)
        # cv2.waitKey()

        key_pts.append((int(x + w / 2), int(y + h)))

    return key_pts, bbox_st


def est_water_boundary(seg_dir, out_dir, key_pts, homo_mat):
    seg_list = sorted(glob(os.path.join(seg_dir, '*.png')))
    out_list = sorted(glob(os.path.join(out_dir, '*.png')))

    water_level_px = np.zeros(len(key_pts))

    for i in trange(len(key_pts)):

        seg = load_image_in_PIL(seg_list[i], 'P')
        seg = np.array(seg)
        seg = cv2.warpPerspective(seg, homo_mat, (seg.shape[1], seg.shape[0]))

        if i > 0:
            water_level_px[i] = water_level_px[i - 1]

        for y in range(key_pts[i][1] + 1, seg.shape[0]):
            if seg[y][key_pts[i][0]] == water_label_id:
                water_level_px[i] = y - key_pts[i][1]

                overlay = cv2.imread(out_list[i])
                cv2.line(overlay, key_pts[i], (key_pts[i][0], y), (200, 0, 0), 2)
                cv2.imwrite(out_list[i], overlay)
                # cv2.imshow('overlay', overlay)
                # cv2.waitKey()
                break

    return water_level_px


def get_time_arr(img_folder):
    img_list = os.listdir(img_folder)
    img_list.sort(key=lambda x: (len(x), x))

    time_arr = []

    for img_name in img_list:
        # year = img_name[-17:-13]
        # mon = img_name[-20:-18]
        # day = img_name[-23:-21]
        # timestamp = f'{year}-{mon}-{day} ' + img_name[-12:-4].replace('-', ':')
        # timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

        timestamp = datetime.strptime(img_name[:-4], '%Y-%m-%d-%H-%M-%S')

        time_arr.append(timestamp)

    return time_arr


def est_waterlevel(img_dir, seg_dir, out_dir, recalib_flag=False, reref_flag=False, sample_times=1):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Calibrate the video
    homo_mat_path = os.path.join(out_dir, 'homo_mat.txt')
    homo_mat = get_video_homo(img_dir, homo_mat_path, recalib_flag)

    time_arr = get_time_arr(img_dir)
    time_arr_path = os.path.join(out_dir, 'time_arr.npy')
    # print(time_arr)
    np.save(time_arr_path, np.array(time_arr))

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ref_bbox_path = os.path.join(out_dir, 'ref_bbox.npy')
    ref_bbox = []
    if not reref_flag and os.path.exists(ref_bbox_path):
        print('Load bbox of the reference objects.', ref_bbox_path)
        ref_bbox = np.load(ref_bbox_path)
        sample_times = ref_bbox.shape[0]

    for sample_iter in range(sample_times):

        print(f'Estimate water level by reference {sample_iter}')

        # Get points of reference objs
        if reref_flag or sample_iter == len(ref_bbox):
            key_pts, bbox_st = track_object(img_dir, seg_dir, out_dir, homo_mat, sample_iter)
            ref_bbox.append(bbox_st)
        else:
            key_pts, bbox_st = track_object(img_dir, seg_dir, out_dir, homo_mat, sample_iter,
                                            tuple(ref_bbox[sample_iter]))

        print('Calculate water level.')
        water_level_px = est_water_boundary(seg_dir, out_dir, key_pts, homo_mat)
        print('Init height in px:', water_level_px[0])
        water_level_px = -(water_level_px - water_level_px[0])
        # print(water_level_px)

        if sample_iter == 0:
            water_level_px_all = water_level_px
        else:
            water_level_px_all = np.vstack((water_level_px_all, water_level_px))

        data_n = len(water_level_px)
        ax.plot(time_arr[:data_n], water_level_px, '+', label=f'By ref {sample_iter+1} (px)')
        # plt.gca().xaxis.set_major_formatter(time_fmt)
        # plt.show()

        # water_level_path = os.path.join(water_level_folder, f'ref{sample_iter}.png')
        # plt.savefig(water_level_path, dpi=300)

    np.save(ref_bbox_path, np.array(ref_bbox))

    if sample_times == 1:
        # ax.plot(time_arr, water_level_px_all.mean(0), '.', label=f'Avg (px)')
        pass
    else:
        water_level_px_all = np.expand_dims(water_level_px_all, axis=0)
    water_level_path = os.path.join(out_dir, 'water_level_px.npy')
    np.save(water_level_path, water_level_px_all)

    ax.xaxis.set_major_formatter(time_fmt)
    ax.xaxis.set_major_locator(ticker_locator)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)

    water_level_path = os.path.join(out_dir, 'water_level_px_raw.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)

    print('Save water_level_px.npy and water_level_px.png')


def smooth_arr(arr, sigma=1):
    smoothed_arr = gaussian_filter1d(arr, sigma, mode='nearest')
    return smoothed_arr


def plot_hydrograph(out_dir, gt_dir):
    gt_col_id = 2
    print('Load water_level_px.npy and time_arr.npy')

    water_level_path = os.path.join(out_dir, 'water_level_px.npy')
    water_level_px_all = np.load(water_level_path)

    time_arr_path = os.path.join(out_dir, 'time_arr.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)

    if args.smooth:
        water_level_px_all = smooth_arr(water_level_px_all)

    gt_path = os.path.join(gt_dir, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)

    gt_csv.iloc[:, 0] = gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1]
    time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0]) - timedelta(minutes=60)

    water_level_px_avg = water_level_px_all.mean(0, keepdims=True)
    px_to_dist_path = os.path.join(out_dir, 'px_to_dist.txt')
    if args.eval and os.path.exists(px_to_dist_path):
        px_to_dist = np.loadtxt(px_to_dist_path)
    else:
        px_to_dist = fit(water_level_px_avg, gt_csv.iloc[:, gt_col_id], time_arr_eval, time_arr_gt)
        np.savetxt(px_to_dist_path, px_to_dist)

    water_level_px_avg = px_to_dist[:, 0] * water_level_px_avg + px_to_dist[:, 1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], '-', label=f'Groundtruth (ft)')

    # for i in range(water_level_px_all.shape[0]):
    #     ax.plot(time_arr_eval, water_level_px_all[i, :], '.', label=f'By ref {i} (ft)')

    # if water_level_px_all.shape[0] > 1:
    ax.plot(time_arr_eval, water_level_px_avg[0], '+', label=f'Avg (ft)')

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)

    water_level_path = os.path.join(out_dir, 'water_level_ft_all.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)
    # plt.show()
    #
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111)
    #
    # ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], label=f'Groundtruth (ft)')
    #
    # if water_level_px_all.shape[0] > 1:
    #     ax.plot(time_arr_eval, water_level_px_all.mean(0), '.', label=f'Estimated Waterlevel (ft)')
    # else:
    #     ax.plot(time_arr_eval, water_level_px_all[0, :], '.', label=f'By ref 0 (ft)')
    #
    # ax.xaxis.set_major_formatter(time_fmt)
    # ax.xaxis.set_major_locator(ticker_locator)
    # plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=font_size)
    # plt.setp(ax.get_yticklabels(), fontsize=font_size)
    # ax.legend(loc='lower right', fontsize=font_size)
    #
    # water_level_path = os.path.join(out_dir, 'water_level_ft_cmp.png')
    # fig.tight_layout()
    # fig.savefig(water_level_path, dpi=300)


def plot_hydrograph2(out_dir, gt_dir):
    gt_col_id = 2
    print('Load water_level_px.npy and time_arr.npy')

    water_level_path = os.path.join(out_dir, 'water_level_px.npy')
    water_level_px_all = np.load(water_level_path)

    time_arr_path = os.path.join(out_dir, 'time_arr.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)

    if args.smooth:
        water_level_px_all = smooth_arr(water_level_px_all[0])
    water_level_px_avg = water_level_px_all.mean(0, keepdims=True)

    if args.video_name == 'boston_harbor_20190119_20190123_day_s':
        split_time = datetime.strptime('2019-01-21', '%Y-%m-%d')
    elif args.video_name == 'Boston_Tea_Party_Museum_Webcam_2_20200620-20200625_day_s':
        split_time = datetime.strptime('2020-06-19', '%Y-%m-%d')

    selected_item = time_arr_eval < split_time
    not_selected_item = time_arr_eval >= split_time

    time_arr_eval_train = time_arr_eval[selected_item]
    time_arr_eval_test = time_arr_eval[not_selected_item]

    water_level_px_avg_train = water_level_px_avg[:, selected_item]
    water_level_px_avg_test = water_level_px_avg[:, not_selected_item]

    gt_path = os.path.join(gt_dir, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)

    gt_csv.iloc[:, 0] = gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1]
    time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0]) - timedelta(minutes=60)

    px_to_dist_path = os.path.join(out_dir, 'px_to_dist.txt')
    if args.eval and os.path.exists(px_to_dist_path):
        px_to_dist = np.loadtxt(px_to_dist_path)
    else:
        px_to_dist = fit(water_level_px_avg_train, gt_csv.iloc[:, gt_col_id], time_arr_eval_train, time_arr_gt)
        np.savetxt(px_to_dist_path, px_to_dist)

    water_level_px_avg_train = px_to_dist[:, 0] * water_level_px_avg_train + px_to_dist[:, 1]
    water_level_px_avg_test = px_to_dist[:, 0] * water_level_px_avg_test + px_to_dist[:, 1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.plot(time_arr_gt, gt_csv.iloc[:, gt_col_id], '-', label=f'Groundtruth (meter)')

    # for i in range(water_level_px_all.shape[0]):
    #     ax.plot(time_arr_eval, water_level_px_all[i, :], '.', label=f'By ref {i} (ft)')

    # if water_level_px_all.shape[0] > 1:
    ax.plot(time_arr_eval_train, water_level_px_avg_train[0], 'o', markersize=10, label=f'Estimated (train)')
    ax.plot(time_arr_eval_test, water_level_px_avg_test[0], 'o', markersize=10, label=f'Estimated (val)')

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)

    water_level_path = os.path.join(out_dir, 'water_level_ft_all.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)



def plot_hydrograph_houston_buffalo(out_dir, gt_dir):
    gt_col_id = 1
    print('Load water_level_px.npy and time_arr.npy')

    water_level_path = os.path.join(out_dir, 'water_level_px.npy')
    water_level_px = np.load(water_level_path)

    time_arr_path = os.path.join(out_dir, 'time_arr.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)[:len(water_level_px)]

    if args.smooth:
        water_level_px = smooth_arr(water_level_px)

    # water_level_est = water_level_est * 56.69 + 10.72  # 2.72
    water_level_est = 0.6 * water_level_px + 5

    gt_path = os.path.join(gt_dir, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)
    old_csv = pd.read_csv('/Ship01/Dataset/VOS/water/WaterlevelGT/houston_buffalo_20170825_20170901_s/water_level.csv')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    data_n = len(water_level_est)
    ax.plot(time_arr_eval[:data_n], water_level_est, '+', label=f'Waterlevel (ft)', lw=2)

    old_n = data_n
    time_arr_old = pd.to_datetime(gt_csv.iloc[:old_n, 2])
    ax.plot(time_arr_old, old_csv.iloc[:old_n, 3], '*', label=f'Old Waterlevel (ft)')

    gt_n = 264
    time_arr_gt = pd.to_datetime(gt_csv.iloc[:gt_n, 0])
    ax.plot(time_arr_gt, gt_csv.iloc[:gt_n, gt_col_id], '-', markersize=15, label=f'Groundtruth (ft)')

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)

    water_level_path = os.path.join(out_dir, 'water_level_ft_all.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)


def est_lsu_water_level(seg_dir, out_dir, overlay_dir):

    seg_list = sorted(glob(os.path.join(seg_dir, '*.png')))
    overlay_list = sorted(glob(os.path.join(overlay_dir, '*.png')))

    n = len(seg_list)
    gauge_label_id = 2
    water_level_px = np.zeros((2, n))

    for i in trange(len(seg_list)):

        seg = load_image_in_PIL(seg_list[i], 'P')
        seg = np.array(seg)

        if i == 0:
            gauge_pos = (seg == gauge_label_id).nonzero(as_tuple=False)
            gauge_top_pos = (int(gauge_pos[1].mean()), int(gauge_pos[0].min())+3)

        else:
            water_level_px[:, i] = water_level_px[:, i - 1]

        gauge_end_flag = False

        for y in range(gauge_top_pos[1] + 5, seg.shape[0]):

            if not gauge_end_flag and seg[y][gauge_top_pos[0]] != gauge_label_id:
                water_level_px[0, i] = y - gauge_top_pos[1]
                gauge_end_flag = True

            if seg[y][gauge_top_pos[0]] == water_label_id:
                water_level_px[1, i] = y - gauge_top_pos[1]

                overlay = cv2.imread(overlay_list[i])
                cv2.line(overlay, gauge_top_pos, (gauge_top_pos[0], y), (200, 0, 0), 2)
                cv2.imwrite(os.path.join(out_dir, os.path.basename(overlay_list[i])), overlay)
                # cv2.imshow('overlay', overlay)
                # cv2.waitKey()
                break

    water_level_px = water_level_px[:, 0:1] - water_level_px

    print(water_level_px)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.plot(water_level_px[0], 'g-', label=f'Gauge top to gauge bottom')
    ax.plot(water_level_px[1], 'r-', label=f'Gauge top to water top')

    ax.legend(loc='lower right', fontsize=font_size)

    water_level_path = os.path.join(out_dir, 'water_gauge.png')
    fig.tight_layout()
    fig.savefig(water_level_path, dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate water level.')
    parser.add_argument(
        '--video-name', type=str, default='boston_harbor_20190119_20190123_day_s',
        help='Video name.')
    parser.add_argument(
        '--recalib', action='store_true',
        help='Recalibate the video')
    parser.add_argument(
        '--reref', action='store_true',
        help='Re-pick the reference objects in the video')
    parser.add_argument(
        '--samples', type=int, default=1,
        help='Recalibate the video')
    parser.add_argument(
        '--plot', action='store_true',
        help='Recalibate the video')
    parser.add_argument(
        '--eval', action='store_true',
        help='Evaluate the water level in the video.')
    parser.add_argument(
        '--smooth', action='store_true',
        help='Smooth the water level in the video.')
    args = parser.parse_args()

    print('Args:', args)

    # Paths
    img_root = '/Ship01/Dataset/VOS/water'
    seg_root = 'output/AFB-URR_Water_fulltrain'
    out_root = 'output/waterlevel'

    img_dir_name = args.video_name[:args.video_name.index('_label_')]

    img_dir = os.path.join(img_root, 'JPEGImages', img_dir_name)
    seg_dir = os.path.join(seg_root, args.video_name)
    out_dir = os.path.join(out_root, args.video_name)
    gt_dir = os.path.join(img_root, 'WaterlevelGT', img_dir_name)
    overlay_dir = os.path.join('overlay/AFB-URR_Water_fulltrain', args.video_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if 'houston_buffalo_20170825_20170901_s' in args.video_name:
        water_label_id = 1
        enable_tracker = False
    else:
        water_label_id = 1
        enable_tracker = True

    if args.video_name[:3] == 'LSU':
        est_lsu_water_level(seg_dir, out_dir, overlay_dir)
        exit()


    if args.plot:
        if 'houston_buffalo_20170825_20170901_s' in args.video_name:
            plot_hydrograph_houston_buffalo(out_dir, gt_dir)
        else:
            plot_hydrograph2(out_dir, gt_dir)
    else:
        est_waterlevel(img_dir, seg_dir, out_dir, args.recalib, args.reref, args.samples)
