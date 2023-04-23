#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:12:41 2022

@author: Tom Bu
"""

import os
import shutil
import matplotlib.pyplot as plt
import pickle
from shapely.geometry import Polygon
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

import descartes
from shapely import geometry
import numpy as np
from matplotlib.patches import Patch
from shapely.geometry.multipolygon import MultiPolygon 
from utils.offline_detections import iou
from utils.offline_detections import get_3d_points

def get_eval(query_sparse, experiment, localization_check):
    with open(os.path.join(query_sparse, experiment, 'label.pkl'), 'rb') as f:
        gt_label = pickle.load(f)
        #remove added crosswalks in the reference. this is because added crosswalks are crosswalks that don't exist in the reference map, but you need these polygons when you evaluate 
        #an example is the transformed crosswalk in cw_9. the reference doesn't have the crosswalk
        #but in order to evaluate the the prediction is added, we keep the added mark for evaluation
        #this is covered in lines 130-150
        ref_detection = [x for x in gt_label if x['change_type'] != 'added']
        
    if localization_check:
        with open(os.path.join(query_sparse, 'localization_query_det.pkl'), 'rb') as f:
            curr_detection = pickle.load(f)         
    else:
        with open(os.path.join(query_sparse, 'query_det.pkl'), 'rb') as f:
            curr_detection = pickle.load(f)
    
    points = get_3d_points(query_sparse)
    points = points[::5, :]
    
    #inference
    same = []
    removed = []    
    rm_ref = []
    rm_pred = []
    for i, gt_poly in enumerate(ref_detection):
        seen = False
        for j, pred_poly in enumerate(curr_detection):
            if seen:
                continue
            if iou(MultiPolygon(gt_poly['poly']), pred_poly) > 0.1:
                seen = True
                same.append(gt_poly)
                rm_ref.append(i)
                rm_pred.append(j)
        if seen == False:
            #if it's not detected and we're using the sensor and the polygon is hand labelled, then we can call it the same
            if 'sensor' in experiment and gt_poly['source'] == 'map':
                same.append(gt_poly)
                rm_ref.append(i)
            else:
                #we only indicate its a removed crosswalk if we observe it. we don't penalize it if we don't think we can see it
                if '_obs' in experiment: 
                    if gt_poly['observed']:
                        removed.append(gt_poly)        
                        rm_ref.append(i)
                    else:
                        rm_ref.append(i)
                else:
                    removed.append(gt_poly)        
                    rm_ref.append(i)                    
    
    ref_detection = [x for i, x in enumerate(ref_detection) if i not in rm_ref]
    curr_detection = [x for i, x in enumerate(curr_detection) if i not in rm_pred]

    added = []
    for pred_poly in curr_detection:
        seen = False
        for gt_poly in ref_detection:
            if seen:
                continue
            if iou(pred_poly, MultiPolygon(gt_poly['poly'])) > 0.1:
                seen = True
                same.append(gt_poly)
        if seen == False:
            added.append(pred_poly)
    fig = plt.figure(figsize=(9, 16))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.scatter(points[:, 0], points[:, 1], s = 0.01, color = 'white')

    for poly in added:
        ax.add_patch(descartes.PolygonPatch(poly, fc = 'b'))
    for poly in removed:
        ax.add_patch(descartes.PolygonPatch(MultiPolygon(poly['poly']), fc = 'r'))
    for poly in same:
        ax.add_patch(descartes.PolygonPatch(MultiPolygon(poly['poly']), fc = 'g'))        
    r = Patch(facecolor='b', label='added')
    b = Patch(facecolor='r', label='removed')
    g = Patch(facecolor='g', label='unchanged')
    ax.legend(handles=[r, g, b])
    

    plt.savefig(os.path.join(query_sparse, experiment, 'change_pred.jpg'))
    
    #save another figure to show the difference between the detections  
    for poly in ref_detection:
        ax.add_patch(descartes.PolygonPatch(MultiPolygon(poly['poly']), fc = 'y'))        
    for poly in curr_detection:
        ax.add_patch(descartes.PolygonPatch(poly, fc = 'm'))                  
    m = Patch(facecolor='m', label='query')
    y = Patch(facecolor='y', label='ref')
    ax.legend(handles=[r, g, b, m, y])
    
    plt.savefig(os.path.join(query_sparse, experiment, 'change_debug.jpg'))
    





    #go through all the cases possible: ref and query can match. query can be a false positive. ref is not matched
    #check the ID to determine the correctness, for same and removed
    #for added, check the IOU
    #the rest are all incorrect
    #these are true positive
    added_matched = []
    #these are false positive detections
    added_unmatched = []
    added_ref = []
    ref_detection = [x for x in gt_label if x['change_type'] == 'added']
    for i, pred_poly in enumerate(added):
        seen = False
        for j, gt_poly in enumerate(ref_detection):
            if seen:
                continue
            if iou(MultiPolygon(gt_poly['poly']), pred_poly) > 0.1:
                seen = True
                added_matched.append(gt_poly)
                added_ref.append(j)
        if seen == False:
            added_unmatched.append({'poly':pred_poly, 'change': False, 'change_type': 'same'})
    added = added_matched + added_unmatched
    #if none of the gt added are detected, then the prediction is essentially no change
    #these are false negative, if it's visible
    if 'obs' in experiment:
        same_unmatched = [x for i, x in enumerate(ref_detection) if i not in added_ref and x['observed']]
    else:
        same_unmatched = [x for i, x in enumerate(ref_detection) if i not in added_ref]
   
    same.extend(same_unmatched)
     
    #matched are same
    #references unmatched are either removed or same (if it's a map)
    #sensors unmatched are added
    
       
    #evaluate
    evaluate = {'any_change_existing_cw':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                'any_change_cw':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                'any_change_intersection':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                }
    specific_cw = {}
    for cw in gt_label:
        specific_cw[cw['id']] = {'tp':0, 'fp': 0, 'tn': 0, 'fn':0}
    evaluate['specific_cw'] = specific_cw
    
    #the lists "added", "same" and "removed" are the predictions. The gt is contained inside
    predictions = {'added':added, 'same':same, 'removed':removed}

    for change_prediction, li in predictions.items():
        for cw in li:
            if 'id' in cw:
                if change_prediction in ['added', 'removed']:
                    if cw['change_type'] == change_prediction:
                        evaluate['specific_cw'][cw['id']]['tp'] += 1
                    else:
                        evaluate['specific_cw'][cw['id']]['fp'] += 1
                else:
                    if cw['change_type'] == change_prediction:
                        evaluate['specific_cw'][cw['id']]['tn'] += 1
                    else:
                        evaluate['specific_cw'][cw['id']]['fn'] += 1                    
        

    #change for existing cw: tp, fp, tn, fn, 
    for change_prediction, li in predictions.items():
        for cw in li:
            if 'id' in cw:
                if change_prediction in ['added', 'removed']:
                    if cw['change_type'] == change_prediction:
                        evaluate['any_change_existing_cw']['tp'] += 1
                    else:
                        evaluate['any_change_existing_cw']['fp'] += 1
                else:
                    if cw['change_type'] == change_prediction:
                        evaluate['any_change_existing_cw']['tn'] += 1
                    else:
                        evaluate['any_change_existing_cw']['fn'] += 1                    
        
    #change in general: tp, fp, tn, fn, 
    for change_prediction, li in predictions.items():
        for cw in li:
            if change_prediction in ['added', 'removed']:
                if cw['change_type'] == change_prediction:
                    evaluate['any_change_cw']['tp'] += 1
                else:
                    evaluate['any_change_cw']['fp'] += 1
            else:
                if cw['change_type'] == change_prediction:
                    evaluate['any_change_cw']['tn'] += 1
                else:
                    evaluate['any_change_cw']['fn'] += 1                    
         
    #intersection level: tp, fp, tn, fn
    change_li = [x for x in gt_label if x['change_type'] != 'same']
    if len(added + removed) > 0:
        if len(change_li) > 0:
            evaluate['any_change_intersection']['tp'] += 1
        else:
            evaluate['any_change_intersection']['fp'] += 1
    else:
        if len(change_li) > 0:
            evaluate['any_change_intersection']['fn'] += 1
        else:
            #if add, removed, same are all empty, then we just ignore this intersection
            if len(same) > 0:
                evaluate['any_change_intersection']['tn'] += 1        

    
    with open(os.path.join(query_sparse, experiment, 'change.pkl'), "wb") as poly_file:
        pickle.dump(evaluate, poly_file, pickle.HIGHEST_PROTOCOL)

