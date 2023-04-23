#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:38:20 2022

@author: Tom Bu
"""
import os
import shutil
from utils.dynamic_obj_mask import generate_masks
from utils.automate_colmap import sfm
# from utils.automate_estimate_grnd_plane_segmentation_aligned import estimate_ground_plane
from utils.automate_estimate_grnd_plane_segmentation_aligned_cw import estimate_ground_plane
from utils.offline_detections import get_det
from utils.automate_dense_reconstruction import dense_reconstruction


class dated_config:
    def __init__(self, day = None):
        self.model_path = "./model/crosswalk_detector/model_final.pth"

        self.sfm_path = day + '/'
        self.sfm_img_path = self.sfm_path + '{}/images'
        self.sfm_ref_path = self.sfm_path + '{}/images/camera3'
        self.sfm_mask_path = self.sfm_path + '{}/mask'
        self.sfm_sparse_path = self.sfm_path + '{}/sparse'
        self.sfm_estimated_ground_output = self.sfm_path + '{}/sparse/plane_segmentation'
        self.sfm_ground_seg_path = self.sfm_path + '{}/ground_mask'

li = [
    'cw_0_267',
    'cw_0_87',
    'cw_10_210',
    'cw_10_87',
    'cw_11_213',
    'cw_11_33',
    'cw_12_210',
    'cw_12_31',
    'cw_13_208',
    'cw_13_30',
    'cw_14_210',
    'cw_14_22',
    'cw_15_255',
    'cw_15_31',
    'cw_16_269',
    'cw_16_73',
    'cw_18_260',
    'cw_19_251',
    'cw_3_71',
    'cw_4_72',
    'cw_5_171',
    'cw_5_36',
    'cw_6_219',
    'cw_6_327',
    'cw_7_214', 
    'cw_7_291', 
    'cw_8_105',
    'cw_8_290',
    'cw_9_109',
    'cw_9_217'
 ]

config = dated_config('2022_10_16_offline_map_query')
for group in li:
    os.makedirs(os.path.join(config.sfm_path, group), exist_ok=True)
   try:
        generate_masks(config.sfm_img_path.format(group), config.sfm_mask_path.format(group))
        sfm(config.sfm_img_path.format(group), config.sfm_mask_path.format(group),  config.sfm_sparse_path.format(group))
        estimate_ground_plane(config.sfm_sparse_path.format(group), config.sfm_img_path.format(group), config.sfm_estimated_ground_output.format(group), config.sfm_ground_seg_path.format(group), config.model_path, config.sfm_drivable_area_seg_path.format(group))
        get_det(config.sfm_estimated_ground_output.format(group), config.sfm_estimated_ground_output.format(group), config.sfm_img_path.format(group), config.model_path, debug = True)
        # dense_reconstruction(config.sfm_path + group)
    except:
        print('failed {}'.format(group))