# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from utils.utils import get_subdir
from detectron2.utils.visualizer import _PanopticPrediction

from tqdm import tqdm


def init_model():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def get_mask(predictor, im_path, cfg):
    im = cv2.imread(im_path)
    
    

    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    
    
    pred = _PanopticPrediction(panoptic_seg.to("cpu"), segments_info, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    
    classes = np.array([x[1]['category_id'] for x in pred.instance_masks()])
    instances = np.array([x[0] for x in pred.instance_masks()])
    thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    
    idx = (classes == thing_classes.index('car')) | (classes == thing_classes.index('person')) | (classes == thing_classes.index('truck')) | (classes == thing_classes.index('bicycle')) | (classes == thing_classes.index('motorcycle')) | (classes == thing_classes.index('bus'))
    thing_instances = instances[idx]
    
    classes = np.array([x[1]['category_id'] for x in pred.semantic_masks()])
    instances = np.array([x[0] for x in pred.semantic_masks()])
    stuff_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
    
    idx = (classes == stuff_classes.index('sky'))
    stuff_instances = instances[idx]
    
    if len(thing_instances) == 0 and len(stuff_instances) == 0:
        pred_mask = np.ones_like(im) * 255
    else:
        if len(thing_instances) > 0 and len(stuff_instances) > 0:
            instances = np.concatenate([thing_instances, stuff_instances], axis = 0)
        elif len(thing_instances) == 0:
            instances = stuff_instances
        elif len(stuff_instances) == 0:
            instances = thing_instances
        
        pred_mask = instances[0]
        for instance in instances:
            pred_mask = pred_mask | instance
    
        pred_mask = pred_mask.astype(int)
        pred_mask = 1 - pred_mask
        pred_mask = pred_mask * 255
        pred_mask = np.stack([pred_mask, pred_mask, pred_mask], axis = 2)
    return pred_mask

def recursion_thru_dir(parent, predictor, cfg, args):
    '''
    create masks for all images in the directory and search through subdirectories if present

    '''
    li = os.listdir(os.path.join(args.path, parent))
    li = sorted(li)
    li = [x for x in li if x.split('.')[-1] in ['jpg', 'png', 'jpeg']]
    for i, file in enumerate(tqdm(li)):
        img_path = os.path.join(args.path, parent, file)
        mask = get_mask(predictor, img_path, cfg)
        dst = os.path.join(args.maskpath, parent, file + '.png')
        cv2.imwrite(dst, mask)

    subdir_li = get_subdir(os.path.join(args.path, parent))
    for subdir in subdir_li:
        os.makedirs(os.path.join(args.maskpath, os.path.join(parent, subdir)), exist_ok=True)    
        recursion_thru_dir( os.path.join(parent, subdir), predictor, cfg, args)
                       
class Argument():
    def __init__(self, path, maskpath):
        self.path = path
        self.maskpath = maskpath

def generate_masks(path, maskpath):
    args = Argument(path, maskpath)

    os.makedirs(args.maskpath, exist_ok=True)    
    cfg, predictor = init_model()
    recursion_thru_dir('', predictor, cfg, args)