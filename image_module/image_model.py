import os
import torch
import cv2
import numpy as np
from torchvision.transforms import functional as TF
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor

from .predictor import VisualizationDemo
from .PointRend.point_rend import add_pointrend_config
from .WaterSeg.WaterNet import WaterNetV1

import myutils


class ImageBasedModel:

    def __init__(self, user_config, device, water_model_cp_path):

        cfg_obj, cfg_skeleton = self.setup_cfg(user_config)
        metadata_obj = MetadataCatalog.get(cfg_obj.DATASETS.TEST[0])
        metadata_skeleton = MetadataCatalog.get(cfg_skeleton.DATASETS.TEST[0])
        self.metadata = metadata_obj
        self.metadata.set(**{
            'keypoint_connection_rules': metadata_skeleton.keypoint_connection_rules,
            'keypoint_flip_map': metadata_skeleton.keypoint_flip_map,
            'keypoint_names': metadata_skeleton.keypoint_names
        })

        self.obj_model = DefaultPredictor(cfg_obj)
        self.skeleton_model = DefaultPredictor(cfg_skeleton)
        self.device = device
        self.water_thres = 0.4
        self.viz_colors = [(0, 0, 0), (0, 0, 255)]  # RGB order

        self.water_net_v1 = WaterNetV1().to(device).eval()

        if os.path.isfile(water_model_cp_path):
            print(f'Load checkpoint {water_model_cp_path}')
            checkpoint = torch.load(water_model_cp_path)
            self.water_net_v1.load_state_dict(checkpoint['water_net_v1'])
        else:
            raise IOError(f'Checkpoint does not exist! Path {water_model_cp_path}')

    def seg_img(self, img):

        with torch.no_grad():
            pred_obj = self.obj_model(img)

        visualizer = Visualizer(img, self.metadata, instance_mode=ColorMode.IMAGE)
        instances = pred_obj['instances'].to(torch.device('cpu'))
        viz = visualizer.draw_instance_predictions(predictions=instances)

        water_mask, water_by_img = self.seg_water(img, pred_obj)

        class_list = pred_obj['instances'].pred_classes
        pred_skeleton = None
        if (class_list == 0).nonzero(as_tuple=False).shape[0] > 0:
            pred_skeleton = self.skeleton_model(img)
            for keypoints_per_instance in pred_skeleton['instances'].pred_keypoints:
                visualizer.draw_and_connect_keypoints(keypoints_per_instance)
            viz = visualizer.output

        seg_by_img = myutils.add_overlay(viz.get_image(), water_mask, self.viz_colors)

        viz_dict = {
            'water_by_img': water_by_img,
            'seg_by_img': seg_by_img,
        }

        seg_res = {
            'pred_obj': pred_obj,
            'pred_skeleton': pred_skeleton,
            'water_by_img': water_mask
        }

        return seg_res, viz_dict

    def seg_imgs(self, img_list):

        res_list = []
        for img in img_list:
            obj_res, water_res = self.seg_img(img)
            res_list.append([obj_res, water_res])

        return res_list

    def seg_water(self, img, pred_obj):

        img_tensor = TF.to_tensor(img[:, :, ::-1].copy()).to(self.device)
        non_water_mask = self.mask_background(pred_obj)

        mean = [0.485, 0.456, 0.406]  # RGB order
        std = [0.229, 0.224, 0.225]
        img_tensor = TF.normalize(img_tensor, mean, std).unsqueeze(0)

        with torch.no_grad():
            seg_res = self.water_net_v1(img_tensor).detach()

        water_mask = seg_res[0, 0]
        water_mask[non_water_mask] = 0
        water_mask = water_mask.cpu().numpy()
        water_mask = (np.squeeze(water_mask) > self.water_thres).astype(np.int32)
        water_by_img = myutils.add_overlay(img, water_mask, self.viz_colors)

        return water_mask, water_by_img

    def mask_background(self, predictions, img=None):

        instance_n, h, w = predictions['instances'].pred_masks.shape

        non_water_mask = torch.zeros((h, w), dtype=torch.bool).to(self.device)
        for i in range(instance_n):
            non_water_mask = torch.bitwise_or(non_water_mask, predictions['instances'].pred_masks[i])

        # non_water_mask = non_water_mask.unsqueeze(0).expand(3, -1, -1)
        # zero_ele = torch.zeros(1).to(self.device)
        # img = torch.where(mask_all, img, zero_ele)

        # return img
        return non_water_mask

    @staticmethod
    def setup_cfg(user_config):
        # load config from file and command-line arguments
        obj_cfg = get_cfg()
        add_pointrend_config(obj_cfg)  # add pointrend's default config
        obj_cfg.merge_from_file(user_config['config_file_obj'])
        obj_cfg.merge_from_list(user_config['opts_obj'])

        # Set score_threshold for builtin models
        obj_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = user_config['conf_thres_obj']
        obj_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = user_config['conf_thres_obj']
        obj_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = user_config['conf_thres_obj']
        obj_cfg.freeze()

        skeleton_cfg = get_cfg()
        skeleton_cfg.merge_from_file(user_config['config_file_skeleton'])
        skeleton_cfg.merge_from_list(user_config['opts_skeleton'])

        # Set score_threshold for builtin models
        skeleton_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = user_config['conf_thres_skeleton']
        skeleton_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = user_config['conf_thres_skeleton']
        skeleton_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = user_config['conf_thres_skeleton']
        skeleton_cfg.freeze()

        return obj_cfg, skeleton_cfg
