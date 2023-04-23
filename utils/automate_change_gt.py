#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:26:32 2022

@author: Tom Bu
"""

import os
from datetime import datetime
import pickle

#labels for when crosswalk exist
labels = {
           0: [['09/19/19', '09/19/24']],  #cw 0
           1: [['09/19/19', '09/19/24']],  #cw 0
           2: [['09/19/19', '09/19/24']],  #cw 10
           3: [['09/19/19', '09/19/24']],  #cw 10
           4: [['09/19/19', '09/19/24']],  #cw 10
           5: [['09/19/19', '09/19/24']],  #cw 10
           6: [['09/19/19', '09/19/24']],  #cw 10
           7: [['09/19/19', '09/19/24']],  #cw 10
           8: [['09/19/19', '09/19/24']],  #cw 10
           9: [['09/19/19', '09/19/24']],  #cw 10
          10: [['09/19/19', '09/19/24']],
          11: [['09/19/19', '09/19/24']],
          12: [['09/19/19', '09/19/24']],
          13: [['09/19/19', '09/19/24']],
          14: [['09/19/19', '09/19/24']],
          15: [['09/19/19', '09/19/24']],
          16: [['09/19/19', '09/19/24']],
          17: [['09/19/19', '09/19/24']],
          18: [['09/19/19', '09/19/24']],
          19: [['09/19/19', '09/19/24']],
          20: [['09/19/19', '09/19/24']],
          21: [['09/19/19', '09/19/24']],
          22: [['09/19/19', '09/19/24']],
          23: [['09/19/19', '09/19/24']],
          24: [['09/19/19', '09/19/24']],
          25: [['09/19/19', '09/19/24']],
          26: [['09/19/19', '09/19/24']],
          27: [['09/19/19', '09/19/24']],
          28: [['09/19/19', '09/19/24']],
          29: [['09/19/19', '09/19/24']],
          30: [['09/19/19', '09/19/24']],
          31: [['09/19/19', '09/19/24']],
          32: [['09/19/19', '05/20/22'], ['09/08/22', '09/19/24']], #change #cw14
          33: [['09/19/19', '05/20/22']], #change #cw14
          34: [['09/19/19', '05/20/22']], #change #cw14
          35: [['09/19/19', '05/20/22'], ['09/08/22', '09/19/24']], #change #cw14
          36: [['09/19/19', '09/19/24']],
          37: [['09/19/19', '09/19/24']],
          38: [['09/19/19', '09/19/24']],
          39: [['09/19/19', '09/19/24']],
          40: [['09/19/19', '09/19/24']],
          41: [['09/19/19', '09/19/24']],
          42: [['09/19/19', '09/19/24']],
          43: [['09/19/19', '09/19/24']],
          44: [['09/19/19', '09/19/24']],
          45: [['09/19/19', '09/19/24']],
          46: [['09/19/19', '09/19/24']],
          47: [['09/19/19', '09/19/24']],
          48: [['09/19/19', '09/19/24']],
          49: [['09/19/19', '09/19/24']],
          50: [['09/19/19', '09/19/24']],
          51: [['09/19/19', '05/20/22'], ['07/10/22', '09/19/24']], #change cw 18
          52: [['09/19/19', '09/19/24']],
          53: [['09/19/19', '05/20/22'], ['06/20/22', '09/19/24']], #change cw 19
          54: [['09/19/19', '05/20/22'], ['06/20/22', '09/19/24']], #change #cw 3
          55: [['09/19/19', '09/19/24']], #cw 3
          56: [['09/19/19', '09/19/24']], #cw 3
          57: [['09/19/19', '05/20/22'], ['06/20/22', '09/19/24']], #change #cw 3
          58: [['09/19/19', '05/20/22'], ['07/10/22', '09/19/24']], #change #cw 4
          59: [['09/19/19', '09/19/24']], #cw 5
          60: [['09/19/19', '09/19/24']], #cw 5
          61: [['09/19/19', '09/19/24']], #cw 5
          62: [['09/19/19', '09/19/24']], #cw5
          63: [['09/19/19', '09/19/24']], #cw 6
          64: [['09/19/19', '09/19/24']], #cw 6
          65: [['09/19/19', '09/19/24']], #cw 6
          66: [['09/19/19', '09/19/24']], #cw 6
          67: [['09/19/19', '09/19/24']], #cw 6
          68: [['09/19/19', '09/19/24']], #cw 6
          69: [['09/19/19', '09/19/24']], #cw 7
          70: [['09/19/19', '09/19/24']], #cw 7
          71: [['09/19/19', '09/19/24']], #cw 7
          72: [['09/19/19', '09/19/24']],
          73: [['09/19/19', '09/19/24']],
          74: [['09/19/19', '09/19/24']],
          75: [['09/19/19', '09/19/24']],
          76: [['09/19/19', '09/19/24']], #cw 8
          77: [['09/19/19', '09/19/24']], #cw 8
          78: [['09/19/19', '09/19/24']], #cw 8
          79: [['09/19/19', '09/19/24']], #cw 8
          80: [['09/19/19', '09/19/24']], #cw 8
          81: [['09/19/19', '09/19/24']], #cw 8
          82: [['09/19/19', '09/19/24']], #cw 8
          83: [['09/19/19', '09/19/24']], #cw 8
          84: [['09/19/19', '09/19/24']], #cw 9
          85: [['09/19/19', '09/19/24']], #cw 9
          86: [['09/19/19', '09/19/24']], #cw 9
          87: [['09/19/19', '09/19/24']], #cw 9
          88: [['09/19/19', '09/19/24']],
          89: [['09/19/19', '09/19/24']],
          90: [['11/01/21', '09/19/24']], #change cw 9
          91: [['11/01/21', '09/19/24']]  #change cw 9
         }

def get_change_gt(output, group = 'cw_0_87', reference_type = 'sensor', pretend_map_is_empty = False, query_time = '09/20/22', ref_time = '09/19/20'):
    """
    reference_type: use 'map' or 'sensor' to use either hand labelled map annotations vs detections from mask rcnn
    pretend_map_is_empty: set to true if to test added crosswalk change detection
    group: which location and heading
    query_time: time of query
    ref: time of reference
    
    return a list of list for the crosswalk polygon, the label, and id
    """
    offline_path = '2022_10_16_offline_map_query/'

    query_time = datetime.strptime(query_time, '%m/%d/%y')
    ref_time = datetime.strptime(ref_time, '%m/%d/%y')
    if query_time < ref_time:
        print('note: reference time is after query time which is unexpected')
    
    
    with open(os.path.join(offline_path, group, 'sparse/plane_segmentation/hand_label.pkl'), 'rb') as f:
        map_data = pickle.load(f)  
    with open(os.path.join(offline_path, group, 'sparse/plane_segmentation/label.pkl'), 'rb') as f:
        sensor_data = pickle.load(f)    
        
    #if we pretend the map is empty in the beginning, then we can't use the sensor detections as reference
    if pretend_map_is_empty:
        reference_type = 'map'
    
    #create a file with id and times they exist
    cw_list = [x['id'] for x in map_data]
    gt = []
    for cw_id in cw_list:
        map_poly = [x['poly'] for x in map_data if x['id'] == cw_id]
        sensor_poly = [x['poly'] for x in sensor_data if x['id'] == cw_id]
            
        query_exist = False
        ref_exist = False
        for exist_range in labels[cw_id]:
            if datetime.strptime(exist_range[0], '%m/%d/%y') <= query_time <= datetime.strptime(exist_range[1], '%m/%d/%y'):
                query_exist = True
            if datetime.strptime(exist_range[0], '%m/%d/%y') <= ref_time <= datetime.strptime(exist_range[1], '%m/%d/%y'):
                ref_exist = True
            if pretend_map_is_empty:
                ref_exist = False
            
        #use the sensor detection as the comparison vs using the map
        if reference_type == 'sensor':
            if len(sensor_poly) == 0 and map_poly:
                ref_poly = map_poly
                poly_source = 'map'
            else:
                ref_poly = sensor_poly
                poly_source = 'sensor'
        else:
            ref_poly = map_poly
            poly_source = 'map'
        change = query_exist ^ ref_exist
        #ignore the case where nothing exists
        if change == False and query_exist == False and ref_exist == False:
            continue
        if change:
            if ref_exist == False and query_exist == True:
                change_type = 'added'
            else:
                change_type = 'removed'
        else:
            change_type = 'same'
        gt.append({'poly': ref_poly[0], 'change': change, 'change_type': change_type, 'id': cw_id, 'source': poly_source})   
    with open(os.path.join(output, 'label.pkl'), "wb") as poly_file:
        pickle.dump(gt, poly_file, pickle.HIGHEST_PROTOCOL)
                