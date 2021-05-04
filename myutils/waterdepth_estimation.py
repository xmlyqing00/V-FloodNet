import json
import numpy as np
import cv2
from scipy.spatial.distance import cdist

import myutils


class WaterdepthEstimation:

    def __init__(self, metadata):

        self.metadata = metadata
        self._KEYPOINT_THRESHOLD = 0.05
        self.keypoint_names = self.metadata.get('keypoint_names')
        self.bottom_region_size = 15
        self.bottom_region_area = 2 * (self.bottom_region_size ** 2)
        self.water_thres = 0.05
        self.d_var = 100

        with open('records/skeleton_depths.json', 'r') as f:
            self.skeleton_depths = json.load(f)

        with open('records/stopsign.json', 'r') as f:
            self.stopsign_meta = json.load(f)

        print(self.keypoint_names)

    def est(self, seg_res, viz_dict, img):

        water_depth = dict()
        self.viz_dict = viz_dict

        if seg_res['pred_skeleton']:
            pred_keypoints = seg_res['pred_skeleton']['instances'].pred_keypoints
            vlist, water_depth_by_skeleton = self.est_by_skeleton(pred_keypoints, seg_res['water_mask'])
            if len(vlist) > 0:
                water_depth['skeleton'] = water_depth_by_skeleton
                water_depth['skeleton_vlist'] = vlist

        pred_classes = seg_res['pred_obj']['instances'].pred_classes
        selected = pred_classes == 11  # stopsign id
        pred_masks = seg_res['pred_obj']['instances'].pred_masks[selected]
        if pred_masks.shape[0] > 0:
            vlist, water_depth_by_stopsign = self.est_by_stopsign(pred_masks, img)
            if len(vlist) > 0:
                water_depth['stopsign'] = water_depth_by_stopsign
                water_depth['stopsign_vlist'] = vlist

        return water_depth, self.viz_dict

    def est_by_skeleton(self, pred_keypoints, water_mask):

        key_centers = []
        key_depths = []

        for keypoints_per_instance in pred_keypoints:

            water_depth_keypoint_name = None
            water_depth_x = None
            water_depth_y = None

            for i, keypoint in enumerate(keypoints_per_instance):
                x, y, prob = keypoint
                if prob < self._KEYPOINT_THRESHOLD:
                    continue

                x, y = int(x), int(y)
                bottom_region_l = x - self.bottom_region_size
                bottom_region_r = x + self.bottom_region_size
                bottom_region_t = y - self.bottom_region_size
                bottom_region_b = y + self.bottom_region_size
                bottom_region = water_mask[bottom_region_t:bottom_region_b, bottom_region_l:bottom_region_r]

                water_ratio = bottom_region.sum() / self.bottom_region_area

                # print(bottom_region.shape, water_ratio, self.keypoint_names[i])

                if water_ratio < self.water_thres:
                    continue

                if not water_depth_keypoint_name or (water_depth_y < y and y - water_depth_y > 3):
                    water_depth_keypoint_name = self.keypoint_names[i]
                    water_depth_x = x
                    water_depth_y = y

            print('==', water_depth_keypoint_name, water_depth_x, water_depth_y)

            if water_depth_keypoint_name:
                key_centers.append([water_depth_x, water_depth_y])
                key_depths.append(self.skeleton_depths[water_depth_keypoint_name])

        h, w = water_mask.shape[:2]
        depth = self.calc_depth(key_centers, key_depths, h, w)

        return key_depths, depth
        #
        # if len(key_centers) == 1:
        #     depth = np.ones((h, w)) * key_depths[0]
        #     return depth
        # else:
        #     key_centers = np.array(key_centers)
        #     key_depths = np.array(key_depths)
        #
        #     p = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2).reshape(-1, 2)
        #
        #     d = cdist(p, key_centers, 'euclidean')
        #     d = np.exp(-d / self.d_var)
        #     d = d / d.sum(axis=1, keepdims=True)
        #
        #     depth = np.multiply(d, key_depths).sum(axis=1).reshape(h, w)
        #
        #     return depth

    def est_by_stopsign(self, pred_masks, img):

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
        #
        # cv2.imshow('img', img)
        # cv2.imshow('grad', img_grad)
        # cv2.imshow('edge', img_edge)
        # cv2.waitKey()

        pred_masks = (pred_masks.cpu().numpy()).astype(np.uint8) * 255

        stopsign_d = []
        stopsign_pt = []
        viz_img = self.viz_dict['viz_img']

        for instance_mask in pred_masks:
            edge_map = cv2.Canny(instance_mask, 75, 200)

            cnts = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts[1], key=cv2.contourArea, reverse=True)

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

            stopsign_vec0 = myutils.normalize(pt_center - lines_vert[:, :2])
            stopsign_vec1 = myutils.normalize(pt_center - pt_bottom).reshape(1, 2)

            cos_sim0 = np.abs(np.multiply(lines_vec, stopsign_vec0).sum(axis=1))
            cos_sim1 = np.abs(np.multiply(lines_vec, stopsign_vec1).sum(axis=1))

            lines_parallel = lines_vert[np.bitwise_and(cos_sim0 > 0.999, cos_sim1 > 0.999)]
            lines_end_flag0 = lines_parallel[:, 1] >= pt_top[1] - 20
            lines_end_flag1 = lines_parallel[:, 3] >= pt_top[1] - 20
            poles = lines_parallel[np.bitwise_and(lines_end_flag0, lines_end_flag1)]

            # viz
            # for x1, y1, x2, y2 in poles:
            #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.imshow('img', img)
            # print(approx)
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
            poles_bottom_bias = abs(poles_bottom_arr - poles_bottom_arr.mean(axis=0)).sum(axis=1)
            poles_bottom_bias_std = poles_bottom_bias.min() * 2
            poles_bottom_arr = poles_bottom_arr[poles_bottom_bias < poles_bottom_bias_std]

            # select topk
            d = myutils.dist(poles_bottom_arr, pt_bottom.reshape(1, 2), axis=1)
            rank_d = np.argsort(d)
            topk = len(poles_bottom_arr) // 2
            poles_bottom_arr = poles_bottom_arr[rank_d[topk:]]

            poles_bottom_pt = poles_bottom_arr.mean(axis=0)
            poles_bottom_d = myutils.dist(poles_bottom_pt, pt_bottom, axis=0)
            cos_ratio = (poles_bottom_pt[1] - pt_bottom[1]) / poles_bottom_d

            # print(poles_bottom_pt)

            px2cm = self.stopsign_meta['size'] / stopsign_h
            pole_d_cm = px2cm * poles_bottom_d
            pole_h_cm = pole_d_cm * cos_ratio

            stopsign_in_water = max(0, self.stopsign_meta['height_urban'] - pole_h_cm)

            print(stopsign_in_water, cos_ratio)

            stopsign_pt.append(poles_bottom_pt)
            stopsign_d.append(stopsign_in_water)

            # viz
            cv2.line(viz_img, tuple(pt_bottom.astype(np.int)), tuple(pt_top.astype(np.int)), (0, 255, 0), 2)
            cv2.line(viz_img, tuple(pt_left.astype(np.int)), tuple(pt_right.astype(np.int)), (0, 255, 0), 2)
            cv2.line(viz_img, tuple(poles_bottom_pt.astype(np.int)), tuple(pt_bottom.astype(np.int)), (0, 0, 255), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey()

        h, w = pred_masks[0].shape[:2]
        depth = self.calc_depth(stopsign_pt, stopsign_d, h, w)
        self.viz_dict['viz_img'] = viz_img

        return stopsign_d, depth

    def calc_depth(self, key_centers, key_depths, h, w):
        if len(key_centers) == 0:
            return None
        elif len(key_centers) == 1:
            depth = np.ones((h, w)) * key_depths[0]
            return depth
        else:
            key_centers = np.array(key_centers)
            key_depths = np.array(key_depths)

            p = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2).reshape(-1, 2)

            d = cdist(p, key_centers, 'euclidean')
            d = np.exp(-d / self.d_var)
            d = d / d.sum(axis=1, keepdims=True)

            depth = np.multiply(d, key_depths).sum(axis=1).reshape(h, w)

            return depth
