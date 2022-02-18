import cv2
import numpy as np
import myutils
import os
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters


rectify_window_name = 'Select 4 pts to get homography matrix'
pts = []
pts_n = 4
loop_flag = True
water_label_id = 1

time_fmt = mdates.DateFormatter('%m-%d %H:%M')
# register_matplotlib_converters()
fontsize = 24
rotation = 45


def mouse_click(event, x, y, flags, param):
    global pts, loop_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv2.EVENT_LBUTTONUP:

        cv2.circle(param, pts[-1], 5, (0, 0, 200), -1)
        cv2.imshow(rectify_window_name, param)

        if len(pts) == pts_n:
            loop_flag = False


def get_video_homo(img_st_path, homo_mat_path):
    # Order: left top, right top, left bottom, right bottom
    if os.path.exists(homo_mat_path):
        print(f'Load homography matrix from {homo_mat_path}')
        homo_mat = np.asmatrix(np.loadtxt(homo_mat_path))
        return homo_mat

    print('Estimate the video homo mat. TopLeft, TopRight, BottomLeft, BottomRight.')

    img_st = cv2.imread(img_st_path)

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


def get_video_ref(ref_img, ref_bbox_path, enable_tracker):

    if os.path.exists(ref_bbox_path):
        print('Load bounding box of the reference object.', ref_bbox_path)
        ref_bbox = np.loadtxt(ref_bbox_path)
    else:
        track_window_name = 'Select A Rect As Reference Obj'
        while True:
            bbox_st = cv2.selectROI(track_window_name, ref_img, fromCenter=False)
            if bbox_st[2] > 0 and bbox_st[3] > 0:
                break
        cv2.destroyWindow(track_window_name)
        ref_bbox = bbox_st
        np.savetxt(ref_bbox_path, np.array(ref_bbox))

    if enable_tracker:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(ref_img, tuple(ref_bbox.astype(np.int)))
    else:
        tracker = None

    return ref_bbox, tracker


def est_by_reference(img_list, water_mask_list, out_dir, enable_tracker, enable_calib):

    if enable_calib:
        homo_mat_path = os.path.join(out_dir, 'homo_mat.txt')
        homo_mat = get_video_homo(img_list[0], homo_mat_path)

    ref_bbox_path = os.path.join(out_dir, 'ref_bbox.txt')
    ref_bbox = None

    waterlevel_list = []
    timestamp_list = []

    viz_dir = os.path.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    for i in trange(len(img_list)):

        img = cv2.imread(img_list[i])
        water_mask = np.asarray(myutils.load_image_in_PIL(water_mask_list[i], 'P'))
        img_size = (img.shape[1], img.shape[0])
        if enable_calib:
            img = cv2.warpPerspective(img, homo_mat, img_size)
            water_mask = cv2.warpPerspective(water_mask, homo_mat, img_size)

        viz_img = myutils.add_overlay(img, water_mask, myutils.color_palette)

        if ref_bbox is None:
            ref_bbox, tracker = get_video_ref(img, ref_bbox_path, enable_tracker)
            waterlevel_list = [0]

        img_name = os.path.basename(img_list[i])[:-4]
        timestamp = datetime.strptime(img_name, '%Y-%m-%d-%H-%M-%S')
        timestamp_list.append(timestamp)
        # Get points of reference objs

        if enable_tracker:
            tracker_flag, bbox = tracker.update(img)
            if tracker_flag:
                ref_bbox = bbox
            else:
                bbox = ref_bbox
        else:
            bbox = ref_bbox

        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 200, 0), 2)

        key_pt = (int(x + w / 2), int(y + h))

        waterlevel_list.append(waterlevel_list[-1])
        for y in range(key_pt[1] + 1, water_mask.shape[0]):
            if water_mask[y][key_pt[0]] == water_label_id:
                waterlevel_list[-1] = y - key_pt[1]
                cv2.line(viz_img, key_pt, (key_pt[0], y), (0, 0, 200), 2)
                break
        
        cv2.imwrite(os.path.join(viz_dir, f'{img_name}.png'), viz_img)

    waterlevel_px = np.array(waterlevel_list[1:])
    waterlevel_px = gaussian_filter1d(waterlevel_px, sigma=2, mode='nearest')
    waterlevel_px = waterlevel_px[0] - waterlevel_px

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.npy')
    np.save(waterlevel_path, waterlevel_px)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(timestamp_list, waterlevel_px, 'o')

    # ax.legend(loc='lower right', fontsize=fontsize)
    ax.xaxis.set_major_formatter(time_fmt)
    ax.set_ylabel('Estimated Water Level (pixel)', fontsize=fontsize)
    # tick_spacing = 1
    # ticker_locator = mdates.MinuteLocator(tick_spacing)
    # ax.xaxis.set_major_locator(ticker_locator)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    # ax.legend(loc='lower right', fontsize=fontsize)

    waterlevel_path = os.path.join(out_dir, 'waterlevel_px.png')
    fig.tight_layout()
    fig.savefig(waterlevel_path, dpi=300)
