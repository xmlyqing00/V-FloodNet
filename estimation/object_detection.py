from tqdm import trange
import cv2
import os
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Instances
import numpy as np
import torch
import json
import warnings

import myutils


stopsign_config = {
    'config_file': 'estimation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml',
    'opts': ['MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'],
    'conf_thres': 0.5,
}
skeleton_config = {
    'config_file': 'estimation/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
    'opts': ['MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl'],
    'conf_thres': 0.7,
}

stopsign_meta = {
    'size': 76.2,  ##91.44,  # 36 inch
    'pole_height': 213.36,  # 7 feet, express highway
}

people_meta = {
    'man_height': 175.4,
    'woman_height': 161.7
}

object_colors = {
    'background': [0, 0, 0],
    'stopsign': [128, 128, 0],
    'skeleton': [0, 128, 128]
}

water_label_id = 1


def draw_instances(img: np.array, instances: Instances):

    for i in range(len(instances)):
        if instances[i].pred_classes != 11:
            continue
        mask = instances[i].pred_masks.squeeze(0).numpy().astype(np.uint8)
        img = myutils.add_overlay(img, mask, object_colors['background'] + object_colors['stopsign'])

    return img


def waterdepth_by_stopsign2(img, instances, water_mask, result_dir, img_name):

    # Constants
    thickness = 6
    template_color = (0, 200, 0)
    submerged_color = (0, 0, 200)
    water_color = (200, 0, 0)

    # Create Template
    pts_n = 8
    degree_step = np.deg2rad(360 / pts_n)
    degree_pos = degree_step / 2
    plate_radius = 50
    plate_center = (200, 100)
    template_size = (400, 400)
    template_plate_height = plate_radius * (2 ** 0.5)
    template_pole_height = template_plate_height / stopsign_meta['size'] * stopsign_meta['pole_height']
    plate_pts = []

    for i in range(pts_n):
        x, y = plate_radius * np.cos(degree_pos), plate_radius * np.sin(degree_pos)
        x, y = x + plate_center[0], y + plate_center[1]
        degree_pos += degree_step
        plate_pts.append((x, y))

    template_plate_pts = np.array(plate_pts)
    template_pole_top = np.mean(template_plate_pts[1:3], axis=0)
    template_pole_bottom = template_pole_top.copy()
    template_pole_bottom[1] += template_pole_height
    template_pole_top, template_pole_bottom = template_pole_top.astype(int), template_pole_bottom.astype(int)

    template_canvas = np.ones((template_size) + (3,)) * 255
    template_plate_pts = template_plate_pts.astype(int)
    for i in range(pts_n):
        cv2.line(template_canvas, template_plate_pts[i], template_plate_pts[(i+1) % pts_n], template_color, thickness)
    cv2.line(template_canvas, template_pole_top, template_pole_bottom, template_color, thickness)

    img_size = img.shape[:2]
    est_canvas = np.ones((img_size) + (3,)) * 255

    submerged_ratio = -1
    waterdepth = -1

    for i in range(len(instances.pred_classes)):
        if instances.pred_classes[i] != 11:  # class index for stopsign
            continue

        edge_map = cv2.Canny(instances.pred_masks[i].numpy().astype(np.uint8) * 255, 75, 200)
        cnts, hierarchy = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        if approx.shape[0] != 8:
            continue
        x, y = approx[:, 0, 0], approx[:, 0, 1]
        x_center, y_center = np.mean(x), np.mean(y)
        r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        angles = np.where((y - y_center) > 0, np.arccos((x - x_center) / r), 2 * np.pi - np.arccos((x - x_center) / r))
        mask = np.argsort(angles)
        x_sorted = x[mask]
        y_sorted = y[mask]
        est_plate_pts = np.float32(np.stack([x_sorted, y_sorted], axis=1))
        template_plate_pts = np.float32(template_plate_pts)

        # trans_mat, status = cv2.findHomography(template_plate_pts, est_plate_pts)
        trans_mat, status = cv2.findHomography(template_plate_pts, est_plate_pts, cv2.RANSAC, 5)
        template_pts = np.concatenate([template_plate_pts, template_pole_top.reshape(1, 2), template_pole_bottom.reshape(1, 2)], axis=0)
        # template_pts2 = np.concatenate([template_pts, np.ones((10, 1))], axis=1)
        # template_pts2_proj = np.matmul(trans_mat, (template_pts2.T)).T
        # template_pts2_proj = template_pts2_proj / template_pts2_proj[:, 2:3]
        template_pts_proj = cv2.perspectiveTransform(template_pts[:, np.newaxis, :], trans_mat)
        template_pts_proj = template_pts_proj.reshape(pts_n + 2, 2).astype(int)
        template_pole_top_proj = template_pts_proj[-2]
        template_pole_bottom_proj = template_pts_proj[-1]
        template_pole_height_proj = myutils.dist(template_pole_top_proj, template_pole_bottom_proj, axis=0)
        template_pts_proj = template_pts_proj[:pts_n]

        viz_img = img.copy()
        for i in range(pts_n):
            cv2.line(viz_img, template_pts_proj[i], template_pts_proj[(i + 1) % pts_n], template_color, thickness)
        viz_img = cv2.line(viz_img, template_pole_top_proj, template_pole_bottom_proj, template_color, thickness)

        # cv2.imshow('viz_img', viz_img)
        # cv2.waitKey()

        # rank_y = np.argsort(template_pts_proj[:, 1], axis=0)
        # est_plate_top = np.mean(approx[rank_y[:2]], axis=0)[0]
        # est_plate_bottom = np.mean(approx[rank_y[-2:]], axis=0)[0]

        # tt_plate = myutils.dist(est_plate_bottom, est_plate_top, axis=0)
        # tt_pole = tt_plate / template_plate_height * template_pole_height

        dir = template_pole_bottom_proj - template_pole_top_proj
        dir = dir / np.linalg.norm(dir)

        est_pole_bottom_water = template_pole_bottom_proj
        for step in range(int(template_pole_height_proj)):
            p = (template_pole_top_proj + dir * step).astype(int)
            if p[0] <= 0 or p[1] <= 0 or p[0] >= img_size[1] or p[1] >= img_size[0]:
                break
            if water_mask[p[1], p[0]] == 1:
                est_pole_bottom_water = p
                break

        submerged_ratio = myutils.dist(est_pole_bottom_water, template_pole_bottom_proj, axis=0) / template_pole_height_proj
        waterdepth = submerged_ratio * stopsign_meta['pole_height']

        est_canvas = cv2.drawContours(est_canvas, cnts, -1, template_color, thickness)
        est_canvas = cv2.line(est_canvas, template_pole_top_proj, template_pole_bottom_proj, template_color,
                              thickness)
        est_canvas = cv2.line(est_canvas, est_pole_bottom_water, template_pole_bottom_proj, submerged_color,
                              thickness)

        template_pole_bottom_water = template_pole_top.copy()
        template_pole_bottom_water[1] += (1 - submerged_ratio) * template_pole_height
        template_pole_top, template_pole_bottom_water = template_pole_top.astype(int), template_pole_bottom_water.astype(int)

        template_pole_bottom_water_left = (template_size[0] // 4, template_pole_bottom_water[1])
        template_pole_bottom_water_right = (template_size[0] * 3 // 4, template_pole_bottom_water[1])
        cv2.line(template_canvas, template_pole_bottom_water, template_pole_bottom, submerged_color, thickness)
        cv2.line(template_canvas, template_pole_bottom_water_left, template_pole_bottom_water_right, water_color,
                 thickness)

        cv2.imwrite(os.path.join(result_dir, f'{img_name}_template.png'), template_canvas)
        cv2.imwrite(os.path.join(result_dir, f'{img_name}_est.png'), est_canvas)
        cv2.imwrite(os.path.join(result_dir, f'{img_name}_pred.png'), viz_img)

        break

    print(submerged_ratio, waterdepth)
    return submerged_ratio, waterdepth


def waterdepth_by_stopsign(img, instances, water_mask, viz_img):

    # Extract poles
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grad = cv2.convertScaleAbs(cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3, scale=0.6))
    ret, img_edge = cv2.threshold(img_grad, 50, 255, cv2.THRESH_BINARY)

    min_line_len = 100
    max_line_gap = 20
    lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 50, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        print('Cannot detect lines in the image. Estimation by stop sign fails.')
        return [], None

    lines = lines.squeeze()
    dir = (abs(lines[:, 0] - lines[:, 2]) + 1) / (abs(lines[:, 1] - lines[:, 3]) + 1)  # dx/dy
    lines_vert = lines[dir < 0.5]
    lines_vec = myutils.normalize(lines_vert[:, 2:] - lines_vert[:, :2])

    # viz
    # for x1, y1, x2, y2 in lines_vert:
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.imshow('grad', img_grad)
    # cv2.imshow('edge', img_edge)
    # cv2.waitKey()

    stopsign_d = []
    stopsign_pt = []
    stopsign_in_waters = []
    raw_data_list = []

    for i in range(len(instances.pred_classes)):
        if instances.pred_classes[i] != 11:  # class index for stopsign
            continue

        edge_map = cv2.Canny(instances.pred_masks[i].numpy().astype(np.uint8) * 255, 75, 200)
        cnts, hierarchy = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        if approx.shape[0] < 8:
            continue

        # stopsign geo
        pt_center = np.mean(approx, axis=0)
        rank_y = np.argsort(approx[:, 0, 1], axis=0)
        pt_top = np.mean(approx[rank_y[:2]], axis=0)[0]
        pt_bottom = np.mean(approx[rank_y[-2:]], axis=0)[0]
        rank_x = np.argsort(approx[:, 0, 0], axis=0)
        pt_left = np.mean(approx[rank_x[:2]], axis=0)[0]
        pt_right = np.mean(approx[rank_x[-2:]], axis=0)[0]

        stopsign_h = myutils.dist(pt_bottom, pt_top, axis=0)
        stopsign_w = myutils.dist(pt_left, pt_right, axis=0)

        # stopsign_h =   pt_bottom[1] - pt_top[1]
        # stopsign_w = pt_right[0] - pt_left[0]

        # stopsign_vec0 = myutils.normalize(pt_center - lines_vert[:, :2])
        stopsign_vec1 = myutils.normalize(pt_center - pt_bottom).reshape(1, 2)

        # direction
        # cos_sim0 = np.abs(np.multiply(lines_vec, stopsign_vec0).sum(axis=1))
        cos_sim1 = np.abs(np.multiply(lines_vec, stopsign_vec1).sum(axis=1))
        # lines_parallel = lines_vert[np.bitwise_and(cos_sim0 > 0.995, cos_sim1 > 0.995)]
        xpd0 = np.bitwise_and(pt_left[0] <= lines_vert[:, 0], lines_vert[:, 0] <= pt_right[0])
        xpd1 = np.bitwise_and(pt_left[0] <= lines_vert[:, 2], lines_vert[:, 2] <= pt_right[0])
        lines_parallel = lines_vert[np.bitwise_and(cos_sim1 > 0.9, np.bitwise_or(xpd0, xpd1))]

        # position
        lines_end_flag0 = lines_parallel[:, 1] >= pt_bottom[1]
        lines_end_flag1 = lines_parallel[:, 3] >= pt_bottom[1]
        lines_parallel = lines_parallel[np.bitwise_or(lines_end_flag0, lines_end_flag1)]

        # dist
        dist0 = abs(lines_parallel[:, 1] - pt_bottom[1]) < stopsign_h * 3
        dist1 = abs(lines_parallel[:, 3] - pt_bottom[1]) < stopsign_h * 3
        poles = lines_parallel[np.bitwise_and(dist0, dist1)]

        # viz
        # for x1, y1, x2, y2 in poles:
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('img', img)
        # tmp = edge_map.copy()
        # cv2.drawContours(tmp, cnts, -1, 255, 3)
        # cv2.imshow('tmp', tmp)
        # cv2.drawContours(edge_map, approx, -1, 255, 3)
        # cv2.imshow('edge', edge_map)
        # cv2.waitKey()

        poles_bottom_arr = []
        thres_faraway = 5 * stopsign_h
        for x1, y1, x2, y2 in poles:

            if y1 < y2:
                if y2 - pt_bottom[1] < thres_faraway:
                    poles_bottom_arr.append([x2, y2])
            else:
                if y1 - pt_bottom[1] < thres_faraway:
                    poles_bottom_arr.append([x1, y1])

        if len(poles_bottom_arr) == 0:
            continue

        poles_bottom_arr = np.array(poles_bottom_arr)

        # remove outliers
        # poles_bottom_bias = abs(poles_bottom_arr - poles_bottom_arr.mean(axis=0)).sum(axis=1)
        # poles_bottom_bias_std = poles_bottom_bias.min() * 2
        # poles_bottom_arr = poles_bottom_arr[poles_bottom_bias < poles_bottom_bias_std]

        # select topk
        # d = myutils.dist(poles_bottom_arr, pt_bottom.reshape(1, 2), axis=1)
        # rank_d = np.argsort(d)
        # topk = len(poles_bottom_arr) // 2
        # poles_bottom_arr = poles_bottom_arr[rank_d[topk:]]

        poles_bottom_pt = poles_bottom_arr.mean(axis=0).astype(np.int32)

        dx = (poles_bottom_pt[0] - pt_bottom[0]) / (poles_bottom_pt[1] - pt_bottom[1])
        pole_x, pole_y = poles_bottom_pt[0], poles_bottom_pt[1]
        for y in range(poles_bottom_pt[1], water_mask.shape[0]):
            if water_mask[y][np.round(pole_x).astype(np.int32)] == water_label_id:
                pole_y = y
                break
            else:
                pole_x += dx

        poles_bottom_pt = np.array([pole_x, pole_y])
        poles_bottom_d = myutils.dist(poles_bottom_pt, pt_bottom, axis=0)
        # cos_ratio = (poles_bottom_pt[1] - pt_bottom[1]) / poles_bottom_d
        raw_data_list.append({
            'pole_top': (*pt_bottom, 1),
            'pole_bottom': (*poles_bottom_pt, 1)
        })

        # print(poles_bottom_pt)

        px2cm = stopsign_meta['size'] / stopsign_h
        pole_h_cm = px2cm * poles_bottom_d
        # pole_h_cm = pole_d_cm * cos_ratio

        stopsign_in_water = max(0, stopsign_meta['height_urban'] - pole_h_cm)
        stopsign_in_waters.append(stopsign_in_water)
        print('Est stopsign in water', stopsign_in_water)

        stopsign_pt.append(poles_bottom_pt)
        stopsign_d.append(stopsign_in_water)

        # viz
        cv2.line(viz_img, tuple(pt_bottom.astype(np.int)), tuple(pt_top.astype(np.int)), (0, 200, 0), 2)
        cv2.line(viz_img, tuple(pt_left.astype(np.int)), tuple(pt_right.astype(np.int)), (0, 200, 0), 2)
        cv2.line(viz_img, tuple(poles_bottom_pt.astype(np.int)), tuple(pt_bottom.astype(np.int)), (0, 0, 200), 2)

        text_pos = pt_bottom.astype(np.int)
        text_pos[0] = max(0, text_pos[0] - 300)
        text_pos[1] = max(0, text_pos[1] + 100)
        text = f'Depth {stopsign_in_water:.1f}cm'
        cv2.putText(viz_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), thickness=3)
        # cv2.imshow('viz_img', viz_img)
        # cv2.waitKey()

    # h, w = pred_masks[0].shape[:2]
    # depth = self.calc_depth(stopsign_pt, stopsign_d, h, w)
    # self.viz_dict['viz_img'] = viz_img

    return stopsign_in_waters, viz_img, raw_data_list


def waterdepth_by_skeleton(pred_keypoints, water_mask, keypoint_names, viz_img):

    key_centers = []
    key_depths = []
    thres_keypoint = 0  # 0.05
    bottom_region_size = 15
    bottom_region_area = 2 * (bottom_region_size ** 2)
    water_thres = 0.05

    raw_data_list = []
    for keypoints_per_instance in pred_keypoints:

        max_depth_keypoint_name = None
        max_depth_x = 0
        max_depth_y = 0
        max_depth = 200

        raw_data_dict = {}
        for i, keypoint in enumerate(keypoints_per_instance):
            x, y, prob = keypoint
            raw_data_dict[keypoint_names[i]] = (x.item(), y.item(), prob.item())

            # if prob < thres_keypoint:
            #     continue

            # x, y = int(x), int(y)
            # bottom_region_l = x - bottom_region_size
            # bottom_region_r = x + bottom_region_size
            # bottom_region_t = y - bottom_region_size
            # bottom_region_b = y + bottom_region_size
            # bottom_region = water_mask[bottom_region_t:bottom_region_b, bottom_region_l:bottom_region_r]
            #
            # water_ratio = bottom_region.sum() / bottom_region_area

            # print(bottom_region.shape, water_ratio, self.keypoint_names[i])

            # if water_ratio < water_thres:
            #     continue

            # if water_mask:
            #     cv2.circle(viz_img, (int(x.item()), int(y.item())), radius=2, color=(0, 200, 0), thickness=2)

            if not max_depth_keypoint_name or (max_depth > skeleton_meta[keypoint_names[i]]):
                max_depth_keypoint_name = keypoint_names[i]
                max_depth_x = x
                max_depth_y = y
                max_depth = skeleton_meta[keypoint_names[i]]

        raw_data_list.append(raw_data_dict)

        if max_depth_keypoint_name and water_mask is not None:
            # key_centers.append([water_depth_x, water_depth_y])
            key_depths.append(max_depth)

            text_pos = (max(0, int(max_depth_x - 250)), max(0, int(max_depth_y - 25)))
            text = f'{max_depth_keypoint_name}: Depth {max_depth:.1f}cm'
            cv2.putText(viz_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), thickness=2)
            print('Est people in water', max_depth_keypoint_name, f'depth {max_depth}cm', 'pos', max_depth_x, max_depth_y)

    return key_depths, viz_img, raw_data_list


def waterdepth_by_skeleton2(instances, img, water_mask, out_dir, img_name):

    img_h, img_w, img_c = img.shape

    for person_idx, pred_box in enumerate(instances.pred_boxes):

        if instances.scores[person_idx] < 0.9:
            continue

        x1, y1, x2, y2 = pred_box.numpy().tolist()

        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        bbox_w = 1.5 * (x2 - x1)
        bbox_h = 1.5 * (y2 - y1)
        radius = max(bbox_w, bbox_h)
        radius = min(min(img_h, img_w), radius) / 2

        left, right = int(center_x - radius), int(center_x + radius)
        top, bottom = int(center_y - radius), int(center_y + radius)
        if left < 0:
            right -= left
            left -= left
        if right >= img_w:
            left -= (right - img_w)
            right = img_w
        if top < 0:
            bottom -= top
            top -= top
        if bottom >= img_h:
            top -= (bottom - img_h)
            bottom = img_h

        person_img = img[top:bottom, left:right, :]
        person_water_mask = water_mask[top:bottom, left:right]
        person_img = cv2.resize(person_img, (224, 224))
        person_water_mask = cv2.resize(person_water_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        out_img_dir = os.path.join(out_dir, 'input')
        os.makedirs(out_img_dir, exist_ok=True)

        out_mask_dir = os.path.join(out_dir, 'mask')
        os.makedirs(out_mask_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_img_dir, f'{img_name}.png'), person_img)
        myutils.save_seg_mask(person_water_mask, os.path.join(out_mask_dir, f'{img_name}.png'))

        # print(img_name)
        # cv2.imshow('mask', water_mask * 255)
        # cv2.imshow('person_mask', person_water_mask * 255)
        # cv2.waitKey()

        break


def predict_boundary(y1: np.array, y2: np.array, resolution):
    y2_bottom = np.median(y2[np.argsort(y2)[-30:]])
    y1_selected = y1 > y2_bottom
    y1 = y1[y1_selected]
    y1_top = np.median(y1[np.argsort(y1)[:10]])
    # print(y2_bottom, y1_top)
    boundary = (y2_bottom + y1_top) // 2

    # Option2, Naive Bayesian Gaussian Distribution
    # pi1 = len(y1) / (len(y1) + len(y2))
    # pi2 = len(y2) / (len(y1) + len(y2))
    # pi1 = pi2 = 1
    # mu1, sigma1 = y1.mean(), y1.std()
    # mu2, sigma2 = y2.mean(), y2.std()
    # sigma12, sigma22 = sigma1 ** 2, sigma2 ** 2
    #
    # a = sigma22 - sigma12
    # b = 2 * (mu2 * sigma12 - mu1 * sigma22)
    # c = mu1 ** 2 * sigma22 - mu2 ** 2 * sigma22 - 2 * sigma12 * sigma22 * np.log((pi1 * sigma2) / (pi2 * sigma1))
    #
    # delta = np.sqrt(b ** 2 - 4 * a * c)
    # x1 = (-b - delta) / (2 * a)
    # x2 = (-b + delta) / (2 * a)
    # print(x1, x2)
    # if 0 < x1 < resolution:
    #     boundary = x1
    # elif 0 < x2 < resolution:
    #     boundary = x2
    # else:
    #     raise ValueError('Can\'t estimate boundary.')

    if np.isnan(boundary):
        return np.NaN, None
    else:
        return boundary.astype(int), y1_selected


def est_by_obj_detection(img_list, water_mask_list, out_dir, opt):

    if opt == 'stopsign':
        user_config = stopsign_config
    elif opt == 'skeleton':
        user_config = skeleton_config
    else:
        raise NotImplementedError(opt)

    # load config from file and command-line arguments
    cfg = get_cfg()
    add_pointrend_config(cfg)  # add pointrend's default config
    cfg.merge_from_file(user_config['config_file'])
    cfg.merge_from_list(user_config['opts'])

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = user_config['conf_thres']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = user_config['conf_thres']
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = user_config['conf_thres']
    cfg.freeze()

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    det_model = DefaultPredictor(cfg)

    waterdepth_list = []
    result_dir = os.path.join(out_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    for i in trange(len(img_list)):
        img_path = img_list[i]
        img_name = os.path.basename(img_path)[:-4]
        img = cv2.imread(img_path)

        try:
            water_mask_path = water_mask_list[i]
            water_mask = np.asarray(myutils.load_image_in_PIL(water_mask_path, 'P'))
        except IndexError:
            water_mask = None
            warnings.warn(f'Water segmentation mask does not exist, {water_mask_list[i]}')

        with torch.no_grad():
            pred_obj = det_model(img)
        # visualizer = Visualizer(img, metadata)
        instances = pred_obj['instances'].to(torch.device('cpu'))

        # if opt == 'stopsign':
        #     # viz_img = draw_instances(img, instances)
        #     viz_img = img
        #     # visualizer.draw_instance_predictions(predictions=instances)
        # else:
        #     for keypoints_per_instance in instances.pred_keypoints:
        #         visualizer.draw_and_connect_keypoints(keypoints_per_instance)
        #         viz_img = visualizer.output.get_image()
        #
        # if water_mask is not None:
        #     viz_img = myutils.add_overlay(viz_img, water_mask, myutils.color_palette)

        if opt == 'stopsign':
            submerge_ratio, waterdepth = waterdepth_by_stopsign2(img, instances, water_mask, result_dir, img_name)
            # raw_data = {
            #     'instances': raw_data_list,
            #     'connection_rules': [('pole_top', 'pole_bottom', (100, 100, 100))]
            # }
            waterdepth_list.append((submerge_ratio, waterdepth))
        else:
            waterdepth_by_skeleton2(instances, img, water_mask, out_dir, img_name)
            # raw_data = {
            #     'instances': raw_data_list,
            #     'connection_rules': metadata.get('keypoint_connection_rules')
            # }
            # cv2.imwrite(os.path.join(out_dir, f'{img_name}.png'), viz_img)

        # pred_res_path = os.path.join(out_dir, img_name + '.json')
        # with open(pred_res_path, 'w') as f:
        #     json.dump(raw_data, f)
    if opt == 'stopsign':
        with open(os.path.join(out_dir, f'waterdepth.txt'), 'w') as f:
            for i in trange(len(img_list)):
                img_name = os.path.basename(img_list[i])[:-4]
                f.write(f'{img_name}\t{waterdepth_list[i][0]:.4f}\t{waterdepth_list[i][1]:.4f}\n')

    elif opt == 'skeleton':

        cmd_str = f'cd /Ship03/Sources/MeshTransformer/ && ' \
                  f'python3.8 /Ship03/Sources/MeshTransformer/metro/tools/inference_bodymesh.py ' \
                  f'--resume_checkpoint=./models/metro_release/metro_3dpw_state_dict.bin ' \
                  f'--image_file_or_path={os.path.abspath(out_dir)}/input/'
        print('Execute', cmd_str)
        os.system(cmd_str)
        print('Execute done.')

        resolution = 224
        with open(os.path.join('./records/template_3Dmesh.txt'), 'r') as f:
            template_3d = np.array(json.load(f))
        template_3d = ((template_3d + 1) * resolution / 2).astype(int)
        template_3d = np.clip(template_3d, 0, resolution - 1)
        template_3d_top = template_3d[:, 1].min()
        template_3d_bottom = template_3d[:, 1].max()
        template_3d_height = template_3d_bottom - template_3d_top

        submerge_ratio_list = []
        for i in trange(len(img_list)):
            img_path = img_list[i]
            img_name = os.path.basename(img_path)[:-4]

            img = cv2.imread(os.path.join(out_dir, 'input', f'{img_name}.png'))
            mask = np.array(myutils.load_image_in_PIL(os.path.join(out_dir, 'mask', f'{img_name}.png'), 'P'))
            overlay = myutils.add_overlay(img, mask, )
            with open(os.path.join(out_dir, 'input', f'{img_name}_pred.txt'), 'r') as f:
                pred_2d = np.array(json.load(f))
            pred_2d = ((pred_2d + 1) * resolution / 2).astype(int)
            pred_2d = np.clip(pred_2d, 0, resolution - 1)

            canvas_est = np.ones((resolution, resolution, 3), np.uint8) * 255
            canvas_template = np.ones((resolution, resolution, 3), np.uint8) * 255
            for j in range(pred_2d.shape[0]):
                cv2.circle(canvas_est, pred_2d[j], 0, [0, 200, 0], 2, lineType=cv2.FILLED)
                cv2.circle(canvas_template, (template_3d[j][0], template_3d[j][1]), 0, [0, 200, 0], 2, lineType=cv2.FILLED)

            water_label = mask[pred_2d[:, 1], pred_2d[:, 0]]
            label_under_water = water_label.nonzero()
            label_above_water = (water_label == 0).nonzero()
            pred_2d_under_water = pred_2d[label_under_water]
            template_2d_under_water = template_3d[label_under_water]
            template_2d_above_water = template_3d[label_above_water]

            # pred_2d_under_water = pred_2d_under_water[np.argsort(pred_2d_under_water[:, 1])]
            for j in range(pred_2d_under_water.shape[0]):
                # cv2.circle(img, pred_2d_under_water[j], 0, [0, 0, 200], 1, lineType=cv2.FILLED)
                cv2.circle(canvas_est, pred_2d_under_water[j], 0, [0, 0, 200], 2, lineType=cv2.FILLED)

            water_boundary, under_water_indices = predict_boundary(template_2d_under_water[:, 1], template_2d_above_water[:, 1], resolution)
            if np.isnan(water_boundary):
                warnings.warn('Cannot estimate the water boundary.')
            else:
                submerge_ratio = 1 - (water_boundary - template_3d_top) / template_3d_height
                print(img_name, 'Estimate water boundary', water_boundary, f'submerge ratio {submerge_ratio:.3f}')
                submerge_ratio_list.append(submerge_ratio)
                # with open(os.path.join(result_dir, f'{img_name}_waterdepth.txt'), 'w') as f:
                #     f.write(str(submerge_ratio))

                water_boundary_left = (int(resolution * 0.25), water_boundary)
                water_boundary_right = (int(resolution * 0.75), water_boundary)
                cv2.line(canvas_template, water_boundary_left, water_boundary_right, (200, 0, 0), 2)

                template_2d_under_water = template_2d_under_water[under_water_indices]
                for j in range(template_2d_under_water.shape[0]):
                    cv2.circle(canvas_template, (template_2d_under_water[j][0], template_2d_under_water[j][1]), 0, [0, 0, 200], 2, lineType=cv2.FILLED)

                # cv2.imshow('img', img)
                # cv2.imshow('canvas_est', canvas_est)
                # cv2.imshow('canvas_template', canvas_template)
                # cv2.imshow('overlay', overlay)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(result_dir, f'{img_name}_est.png'), canvas_est)
                cv2.imwrite(os.path.join(result_dir, f'{img_name}_template.png'), canvas_template)
                cv2.imwrite(os.path.join(result_dir, f'{img_name}_overlay.png'), overlay)

        with open(os.path.join(out_dir, f'waterdepth.txt'), 'w') as f:
            for i in trange(len(img_list)):
                img_name = os.path.basename(img_list[i])[:-4]
                waterdepth = submerge_ratio_list[i] * people_meta['man_height']
                f.write(f'{img_name}\t{submerge_ratio_list[i]:.4f}\t{waterdepth:.4f}\n')

    #
    # def calc_depth(self, key_centers, key_depths, h, w):
    #     if len(key_centers) == 0:
    #         return None
    #     elif len(key_centers) == 1:
    #         depth = np.ones((h, w)) * key_depths[0]
    #         return depth
    #     else:
    #         key_centers = np.array(key_centers)
    #         key_depths = np.array(key_depths)
    #
    #         p = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2).reshape(-1, 2)
    #
    #         d = cdist(p, key_centers, 'euclidean')
    #         d = np.exp(-d / self.d_var)

    #         d = d / d.sum(axis=1, keepdims=True)
    #
    #         depth = np.multiply(d, key_depths).sum(axis=1).reshape(h, w)
    #
    #         return depth