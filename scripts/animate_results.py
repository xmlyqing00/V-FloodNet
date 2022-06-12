import argparse
import cv2
import numpy as np
import os
from glob import glob
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import bisect

time_fmt = mdates.DateFormatter('%m-%d %H:%M')
fontsize = 24
rotation = 45
markersize = 10


def get_parser():
    parser = argparse.ArgumentParser(description='Convert images to videos.')
    parser.add_argument('--img-dir', type=str, default='/Ship01/Dataset/water_v3/test_videos',
                        help='Path to the input image')
    parser.add_argument('--viz-dir', type=str, default='./output/waterlevel',
                        help='Path to the viz image.')
    parser.add_argument('--gt-dir', type=str, default='./records/groundtruth',
                        help='Path to the groundtruth dir.')
    parser.add_argument('--out-dir', type=str, default='./output/animation_videos',
                        help='Path to the input image')
    parser.add_argument('--video-name', type=str, required=True,
                        help='Name of the test video')
    args = parser.parse_args()
    return args

def parse_gt_csv(gt_csv):
    metric_scale = 1
    if 'boston_harbor' in args.video_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1])
        if '20190119_20190123' in args.video_name:
            timestamp_list_gt = timestamp_list_gt - timedelta(minutes=60)
        gt_col_id = 4
        ticker_locator = mdates.HourLocator(interval=6)
        type = 'Water Level'
    elif 'houston' in args.video_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%m/%d/%Y %H:%M')
        gt_col_id = 2
        ticker_locator = mdates.HourLocator(interval=6)
        type = 'Water Level'
    elif 'LSU' in args.video_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], errors='coerce', format='%Y-%m-%d-%H-%M-%S')
        gt_col_id = 1
        type = 'Water Depth'
        if len(timestamp_list_gt) < 15:
            ticker_locator = mdates.MinuteLocator(interval=1)
        else:
            ticker_locator = mdates.MinuteLocator(interval=3)
    else:
        raise NotImplementedError

    gt_val = pd.to_numeric(gt_csv.iloc[:, gt_col_id], 'coerce') * metric_scale

    timestamp_list_gt = np.array(timestamp_list_gt)
    gt_val = np.array(gt_val)

    mask = ~np.isnat(timestamp_list_gt)
    timestamp_list_gt = timestamp_list_gt[mask]
    gt_val = gt_val[mask]

    return timestamp_list_gt, gt_val, ticker_locator, type

def cvt_images_to_video(img_dir,
                        viz_dir,
                        data_path,
                        gt_path,
                        video_path,
                        fps=10):

    img_list = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    img_list.sort(key=lambda x: (len(x), x))
    viz_list = glob(os.path.join(viz_dir, '*.png'))
    viz_list.sort(key=lambda x: (len(x), x))

    print(len(img_list), len(viz_list))
    print(viz_dir)
    assert len(img_list) == len(viz_list) and len(img_list) > 0

    gt_csv = pd.read_csv(gt_path)
    timestamp_list_gt, gt_val, ticker_locator, type = parse_gt_csv(gt_csv)

    metric = 'meters'
    waterlevel = pd.read_csv(data_path, index_col=0)
    timestamp_list_est = np.array(pd.to_datetime(waterlevel.index))
    est_val = np.array(waterlevel[metric])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    y_min = min(est_val.min(), gt_val.min())
    y_max = max(est_val.max(), gt_val.max())
    y_delta = (y_max - y_min) * 0.1
    y_min = y_min - y_delta
    y_max = y_max + y_delta

    if timestamp_list_est[0] < timestamp_list_gt[0]:
        x_min = timestamp_list_est[0]
    else:
        x_min = timestamp_list_gt[0]
    if timestamp_list_est[-1] > timestamp_list_gt[-1]:
        x_max = timestamp_list_est[-1]
    else:
        x_max = timestamp_list_gt[-1]

    for i in trange(1, len(img_list)):
        fig = plt.figure(figsize=(20, 15))

        img = cv2.imread(img_list[i])
        ax = fig.add_subplot(221)
        ax.axis('off')
        ax.set_title('Input Image')
        ax.imshow(img)

        viz = cv2.imread(viz_list[i])
        ax = fig.add_subplot(222)
        ax.axis('off')
        ax.set_title('Segmentation and Estimation')
        ax.imshow(viz)

        ax = fig.add_subplot(212)
        k = max(1, bisect.bisect_left(timestamp_list_gt, timestamp_list_est[i]))
        ax.plot(timestamp_list_gt[:k], gt_val[:k], '^', markersize=markersize, label=f'Groundtruth')
        if 'houston' in args.video_name:
            ax.axhline(y=10.3, linestyle='--')
            ax.plot(timestamp_list_est[:i], est_val[:i], 'o', markersize=markersize // 4, label=f'Estimated {type}')
        else:
            ax.plot(timestamp_list_est[:i], est_val[:i], 'o', markersize=markersize, label=f'Estimated {type}')
        ax.legend(loc='lower right', fontsize=fontsize)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(ticker_locator)
        ax.xaxis.set_major_formatter(time_fmt)
        ax.set_ylabel(f'{type} ({metric})', fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)

        fig.tight_layout()

        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        canvas = canvas.reshape((h, w, 3))

        if i == 1:
            video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        video.write(canvas)
        plt.close(fig)

    video.release()

    print(video_path)


if __name__ == '__main__':
    args = get_parser()

    if 'houston' in args.video_name:
        fps = 120
    elif 'boston_harbor' in args.video_name:
        fps = 15
    elif 'LSU' in args.video_name:
        fps = 3
    else:
        fps = 3

    img_dir = os.path.join(args.img_dir, args.video_name)
    viz_dir = os.path.join(args.viz_dir, f'{args.video_name}_ref', 'viz')
    data_path = os.path.join(args.viz_dir, f'{args.video_name}_ref', 'waterlevel.csv')
    gt_path = os.path.join(args.gt_dir, f'{args.video_name}_gt.csv')
    video_path = os.path.join(args.out_dir, f'{args.video_name}.mp4')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    cvt_images_to_video(img_dir, viz_dir, data_path, gt_path, video_path, fps=fps)