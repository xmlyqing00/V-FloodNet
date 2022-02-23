import cv2
import os
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
import numpy as np
import torch
import json

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
  'size': 76.2,
  'height_urban': 213.36,
  'height_rural': 182.88
}

skeleton_meta = {  # 200 is placeholder
  "nose": 200,
  "left_eye": 200,
  "right_eye": 200,
  "left_ear": 200,
  "right_ear": 200,
  "left_shoulder": 150,
  "right_shoulder": 150,
  "left_elbow": 200,
  "right_elbow": 200,
  "left_wrist": 200,
  "right_wrist": 200,
  "left_hip": 100,
  "right_hip": 100,
  "left_knee": 40,
  "right_knee": 40,
  "left_ankle": 5,
  "right_ankle": 5
}


def waterlevel_by_stopsign(img, instances, water_mask, viz_img):

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
    dir = abs(lines[:, 0] - lines[:, 2]) / abs(lines[:, 1] - lines[:, 3])
    lines_vert = lines[dir < 1]
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
        dist0 = abs(lines_parallel[:, 1] - pt_bottom[1]) < stopsign_h * 5
        dist1 = abs(lines_parallel[:, 3] - pt_bottom[1]) < stopsign_h * 5
        poles = lines_parallel[np.bitwise_or(dist0, dist1)]

        # viz
        # for x1, y1, x2, y2 in poles:
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # cv2.imshow('img', img)
        # viz
        # tmp = edge_map.copy()
        # cv2.drawContours(tmp, cnts, -1, 255, 3)
        # cv2.imshow('mask', instance_mask)
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

        poles_bottom_pt = poles_bottom_arr.mean(axis=0)
        poles_bottom_d = myutils.dist(poles_bottom_pt, pt_bottom, axis=0)
        cos_ratio = (poles_bottom_pt[1] - pt_bottom[1]) / poles_bottom_d
        raw_data_list.append({
            'pole_top': (*pt_bottom, 1),
            'pole_bottom': (*poles_bottom_pt, 1)
        })

        # print(poles_bottom_pt)

        px2cm = stopsign_meta['size'] / stopsign_h
        pole_d_cm = px2cm * poles_bottom_d
        pole_h_cm = pole_d_cm * cos_ratio

        stopsign_in_water = max(0, stopsign_meta['height_urban'] - pole_h_cm)
        stopsign_in_waters.append(stopsign_in_water)
        print('Est stopsign in water', stopsign_in_water, cos_ratio)

        stopsign_pt.append(poles_bottom_pt)
        stopsign_d.append(stopsign_in_water)

        # viz
        cv2.line(viz_img, tuple(pt_bottom.astype(np.int)), tuple(pt_top.astype(np.int)), (0, 200, 0), 2)
        cv2.line(viz_img, tuple(pt_left.astype(np.int)), tuple(pt_right.astype(np.int)), (0, 200, 0), 2)
        cv2.line(viz_img, tuple(poles_bottom_pt.astype(np.int)), tuple(pt_bottom.astype(np.int)), (0, 0, 200), 2)

        text_pos = pt_bottom.astype(np.int)
        text_pos[0] = max(0, text_pos[0] - 200)
        text_pos[1] = max(0, text_pos[1] + 40)
        text = f'Depth {stopsign_in_water:.1f}cm'
        cv2.putText(viz_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), thickness=2)
        # cv2.imshow('viz_img', viz_img)
        # cv2.waitKey()

    # h, w = pred_masks[0].shape[:2]
    # depth = self.calc_depth(stopsign_pt, stopsign_d, h, w)
    # self.viz_dict['viz_img'] = viz_img

    return stopsign_in_waters, viz_img, raw_data_list


def waterlevel_by_skeleton(pred_keypoints, water_mask, keypoint_names, viz_img):

    key_centers = []
    key_depths = []
    thres_keypoint = 0.05
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

            if prob < thres_keypoint:
                continue

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

            if not max_depth_keypoint_name or (max_depth > skeleton_meta[keypoint_names[i]]):
                max_depth_keypoint_name = keypoint_names[i]
                max_depth_x = x
                max_depth_y = y
                max_depth = skeleton_meta[keypoint_names[i]]

        raw_data_list.append(raw_data_dict)

        if max_depth_keypoint_name:
            # key_centers.append([water_depth_x, water_depth_y])
            key_depths.append(max_depth)

            text_pos = (max(0, int(max_depth_x - 250)), max(0, int(max_depth_y - 25)))
            text = f'{max_depth_keypoint_name}: Depth {max_depth:.1f}cm'
            cv2.putText(viz_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), thickness=2)
            print('Est people in water', max_depth_keypoint_name, f'depth {max_depth}cm', 'pos', max_depth_x, max_depth_y)

    return key_depths, viz_img, raw_data_list


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

    for img_path, water_mask_path in zip(img_list, water_mask_list):

        img = cv2.imread(img_path)
        water_mask = np.asarray(myutils.load_image_in_PIL(water_mask_path, 'P'))
        # water_mask = water_mask[..., np.newaxis]

        with torch.no_grad():
            pred_obj = det_model(img)
        visualizer = Visualizer(img, metadata)
        instances = pred_obj['instances'].to(torch.device('cpu'))

        if opt == 'stopsign':
            visualizer.draw_instance_predictions(predictions=instances)
        else:
            for keypoints_per_instance in instances.pred_keypoints:
                visualizer.draw_and_connect_keypoints(keypoints_per_instance)

        viz_img = myutils.add_overlay(visualizer.output.get_image(), water_mask, myutils.color_palette)

        if opt == 'stopsign':
            waterlevels, viz_img, raw_data_list = waterlevel_by_stopsign(img, instances, water_mask, viz_img)
            raw_data = {
                'instances': raw_data_list,
                'connection_rules': [('pole_top', 'pole_bottom', (100, 100, 100))]
            }
        else:
            waterlevels, viz_img, raw_data_list = waterlevel_by_skeleton(instances.pred_keypoints, water_mask, metadata.get('keypoint_names'),  viz_img)
            raw_data = {
                'instances': raw_data_list,
                'connection_rules': metadata.get('keypoint_connection_rules')
            }

        img_name = os.path.basename(img_path)[:-4]
        cv2.imwrite(os.path.join(out_dir, f'{img_name}.png'), viz_img)

        pred_res_path = os.path.join(out_dir, img_name + '.json')
        with open(pred_res_path, 'w') as f:
            json.dump(raw_data, f)

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