import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import argparse
import bisect


time_fmt = mdates.DateFormatter('%m-%d %H:%M')
fontsize = 24
rotation = 45
markersize = 10


def get_parser():
    parser = argparse.ArgumentParser(description='Compare water level.')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of the test video')
    parser.add_argument('--out-dir', default='output/waterlevel',
                        help='A file or directory to save output results.')

    args = parser.parse_args()
    return args

def get_gt_sample(est_time, gt_time, gt_val):
    data_n = est_time.shape[0]
    gt_val_sample = np.zeros(data_n)
    for i in range(data_n):
        k = bisect.bisect_left(gt_time, est_time[i])
        if k == 0:
            gt_val_sample[i] = gt_val[k]
        else:
            r = (est_time[i] - gt_time[k - 1]) / (gt_time[k] - gt_time[k - 1])
            gt_val_sample[i] = gt_val[k - 1] + r * (gt_val[k] - gt_val[k - 1])

    return gt_val_sample


def main(args):

    out_dir = os.path.join(args.out_dir, f'{args.test_name}_{args.opt}')
    gt_dir = './records/groundtruth'

    print('Load waterlevel.csv, gt.csv and px_to_meter.txt')
    waterlevel_path = os.path.join(out_dir, 'waterlevel.csv')
    waterlevel = pd.read_csv(waterlevel_path, index_col=0)

    gt_csv_path = os.path.join(gt_dir, f'{args.test_name}_gt.csv')
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError('Please prepare the groundtruth file like *_gt.csv in ./records/groundtruth')
    gt_csv = pd.read_csv(gt_csv_path)

    px_to_meter_path = os.path.join(gt_dir, f'{args.test_name}_px_to_meter.txt')
    if not os.path.exists(px_to_meter_path):
        raise FileNotFoundError('Please prepare the conversion file like *_px_to_meter.txt in ./records/groundtruth')
    px_to_meter = np.loadtxt(px_to_meter_path)

    if px_to_meter.ndim == 1:
        px_to_meter = px_to_meter[np.newaxis, :]

    metric_scale = 1
    metric = 'meters'
    if 'boston_harbor' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1])
        if '20190119_20190123' in args.test_name:
            timestamp_list_gt = timestamp_list_gt - timedelta(minutes=60)
        gt_col_id = 4
        ticker_locator = mdates.HourLocator(interval=6)
        type = 'Water Level'
    elif 'houston' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], format='%m/%d/%Y %H:%M')
        gt_col_id = 2
        ticker_locator = mdates.HourLocator(interval=6)
        type = 'Water Level'
    elif 'LSU' in args.test_name:
        timestamp_list_gt = pd.to_datetime(gt_csv.iloc[:, 0], errors='coerce', format='%Y-%m-%d-%H-%M-%S')
        gt_col_id = 1
        type = 'Water Depth'
        if len(waterlevel.index) < 15:
            ticker_locator = mdates.MinuteLocator(interval=1)
        else:
            ticker_locator = mdates.MinuteLocator(interval=3)
    else:
        raise NotImplementedError

    tracker_num = px_to_meter.shape[0]
    data_num = len(waterlevel)
    waterlevel_meter = np.zeros((tracker_num, data_num))

    # conversion
    for i in range(tracker_num):
        waterlevel_meter[i] = px_to_meter[i, 0] * waterlevel[f'est_ref{i}_px'] + px_to_meter[i, 1]
    waterlevel[metric] = np.nanmean(waterlevel_meter, axis=0)

    timestamp_list_est = pd.to_datetime(waterlevel.index)
    gt_val = pd.to_numeric(gt_csv.iloc[:, gt_col_id], 'coerce') * metric_scale

    # Calc error
    gt_val_sample = get_gt_sample(timestamp_list_est, timestamp_list_gt, gt_val)
    abs_err = np.absolute(waterlevel[metric] - gt_val_sample)
    abs_err_ratio = np.absolute(abs_err / np.nanmax(gt_val_sample))

    # convert to centimeter
    abs_err *= 100
    abs_err_ratio *= 100
    results = f'Absolute error (cm): mean {abs_err.mean():.3f} std {abs_err.std():.3f} \n' \
              f'Absolute error rate (%): mean {abs_err_ratio.mean():.3f} std {abs_err_ratio.std():.3f} \n'
    waterlevel.to_csv(waterlevel_path)

    print(results)
    with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
        f.write(results)

    # Plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.plot(timestamp_list_gt, gt_val, '^', markersize=markersize, label=f'Groundtruth')

    if 'houston' in args.test_name:
        high_water_val = 10.3
        ax.plot(timestamp_list_est, waterlevel[metric],
                '-', linewidth=markersize//3, label=f'Estimated {type} (Ours)')
        old_col_id = 5
        ax.plot(timestamp_list_est, gt_csv.iloc[:, old_col_id],
                '-', linewidth=markersize//3, label=f'Estimated {type} (Jafari et al.)')
        ax.axhline(y=high_water_val, linestyle='--', linewidth=4)
        ax.text(timestamp_list_est[-1000], high_water_val, 'Observed High Water Mark', va='center', ha='center', backgroundcolor='w', fontsize=fontsize)
        ax.legend(loc='upper right', fontsize=fontsize)
    else:
        ax.plot(timestamp_list_est, waterlevel[metric], 'o', markersize=markersize, label=f'Estimated {type}')
        if 'Boston' in args.test_name:
            ax.legend(loc='upper left', fontsize=fontsize)
        else:
            ax.legend(loc='upper left', fontsize=fontsize)

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    ax.set_ylabel(f'{type} ({metric})', fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    waterlevel_path = os.path.join(out_dir, f'waterlevel_{metric}.png')
    fig.tight_layout()
    fig.savefig(waterlevel_path, dpi=200)

    print(f'Save figure to {waterlevel_path}.')


if __name__ == '__main__':

    _args = get_parser()
    _args.opt = 'ref'

    print(_args)

    main(_args)
