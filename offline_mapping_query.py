#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:38:20 2022

@author: Tom Bu
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.dynamic_obj_mask import generate_masks
from utils.automate_registration import sfm
from utils.offline_detections import get_det
from utils.offline_query_eval import get_eval
from utils.automate_observability import get_observability
from utils.automate_localization_check import get_well_localized_imgs
from utils.automate_change_gt import get_change_gt
from utils.eval_metrics import metrics


class dated_config:
    def __init__(self, day = 'None'):
        self.model_path = "./model/crosswalk_detector/model_final.pth"
        self.sfm_path = '2022_10_16_offline_map_query/'
        self.sfm_query_path = self.sfm_path + '{}/query/' + day + '/'
        self.sfm_img_path = self.sfm_query_path + 'images'
        self.sfm_mask_path = self.sfm_query_path + 'mask'
        self.sfm_sparse_path = self.sfm_query_path + 'sparse'
        self.sfm_sparse_experiment = self.sfm_sparse_path + '/{}'
        self.sfm_offline_path = self.sfm_path + '{}/sparse/plane_segmentation'

class exp_tracker:
    def __init__(self):
        self.metrics_any = {'ref':[], 'obs':[], 'loc':[], 'empty':[], 'precision':[], 'recall':[], 'accuracy':[], 'fpr':[], 'f1':[], 'no_c precision':[]}
        self.metrics_existing = {'ref':[], 'obs':[], 'loc':[], 'empty':[], 'precision':[], 'recall':[], 'accuracy':[], 'fpr':[], 'f1':[], 'no_c precision':[]}
        self.metrics_int = {'ref':[], 'obs':[], 'loc':[], 'empty':[], 'precision':[], 'recall':[], 'accuracy':[], 'fpr':[], 'f1':[], 'no_c precision':[]}


    def update(self, exp, result):
        for metrics, category in [[self.metrics_any,'any_change_cw'], [self.metrics_existing, 'any_change_existing_cw'], [self.metrics_int, 'any_change_intersection']]:
            if exp['ref_polygon'] == 'map':
                metrics['ref'].append('map')
            else:
                metrics['ref'].append('sensor')
            if exp['observable_check']:
                metrics['obs'].append('x')
            else:
                metrics['obs'].append(' ')
            if exp['localization_check']:
                metrics['loc'].append('x')
            else:
                metrics['loc'].append(' ')
            if exp['pretend_empty_map']:
                metrics['empty'].append('x')
            else:
                metrics['empty'].append(' ')
            
            metrics['precision'].append(result[category]['precision'])
            metrics['recall'].append(result[category]['recall'])
            metrics['accuracy'].append(result[category]['accuracy'])
            metrics['fpr'].append(result[category]['fpr'])
            metrics['f1'].append(result[category]['f1'])
            metrics['no_c precision'].append(result[category]['no change precision'])
            
            
    def show(self):
        df = pd.DataFrame.from_dict(self.metrics_any)
        df = df.round(3)
        print(df.to_latex(index=False)) 
        df = pd.DataFrame.from_dict(self.metrics_existing)
        df = df.round(3)
        print(df.to_latex(index=False))         
        df = pd.DataFrame.from_dict(self.metrics_int)
        df = df.round(3)
        print(df.to_latex(index=False)) 

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
experiment_li = [
                  {'ref_polygon':'sensor', 'observable_check':False, 'localization_check':False, 'pretend_empty_map':False}, 
                  {'ref_polygon':'sensor', 'observable_check':True, 'localization_check':False, 'pretend_empty_map':False}, 
                  {'ref_polygon':'sensor', 'observable_check':False, 'localization_check':True, 'pretend_empty_map':False}, 
                  {'ref_polygon':'sensor', 'observable_check':True, 'localization_check':True, 'pretend_empty_map':False},
                 
                   {'ref_polygon':'map', 'observable_check':False, 'localization_check':False, 'pretend_empty_map':False}, 
                   {'ref_polygon':'map', 'observable_check':True, 'localization_check':False, 'pretend_empty_map':False}, 
                   {'ref_polygon':'map', 'observable_check':False, 'localization_check':True, 'pretend_empty_map':False}, 
                  {'ref_polygon':'map', 'observable_check':True, 'localization_check':True, 'pretend_empty_map':False}, 
                 
                  {'ref_polygon':'sensor', 'observable_check':False, 'localization_check':False, 'pretend_empty_map':True}, 
                  {'ref_polygon':'sensor', 'observable_check':True, 'localization_check':False, 'pretend_empty_map':True}, 
                  {'ref_polygon':'sensor', 'observable_check':False, 'localization_check':True, 'pretend_empty_map':True}, 
                  {'ref_polygon':'sensor', 'observable_check':True, 'localization_check':True, 'pretend_empty_map':True}, 
                  {'ref_polygon':'map', 'observable_check':False, 'localization_check':False, 'pretend_empty_map':True}, 
                  {'ref_polygon':'map', 'observable_check':True, 'localization_check':False, 'pretend_empty_map':True}, 
                  {'ref_polygon':'map', 'observable_check':False, 'localization_check':True, 'pretend_empty_map':True}, 
                  {'ref_polygon':'map', 'observable_check':True, 'localization_check':True, 'pretend_empty_map':True}, 
                 ]
#keep track of multiple experiments
tracker = exp_tracker()
failed = []
for x in experiment_li:
    ref_polygon, observable_check, localization_check, pretend_empty_map = x['ref_polygon'], x['observable_check'], x['localization_check'], x['pretend_empty_map']

    experiment = 'base'
    experiment += '_' + ref_polygon
    if observable_check:
        experiment += '_obs'
    if localization_check:
        experiment += '_loc'
    if pretend_empty_map:
        experiment += '_pretendEmptyMap'
    #keep track of one experiment
    m = metrics()
    for group in li:
        days = sorted(os.listdir(os.path.join(dated_config().sfm_path, group, 'query')))
        for date in days:
            config = dated_config(date)

            os.makedirs(os.path.join(config.sfm_sparse_experiment.format(group, experiment)), exist_ok=True)
            
            #if the result exists, continue
            if os.path.exists(os.path.join(config.sfm_sparse_experiment.format(group, experiment), 'change.pkl')):
                print('succeed previously {} {} {}'.format(group, date, experiment))    
                continue            
            try:
                #common across all experiments
                if not os.path.exists(config.sfm_mask_path.format(group)) or len(os.listdir(config.sfm_mask_path.format(group))) != len(os.listdir(config.sfm_img_path.format(group))) or len(os.listdir(config.sfm_mask_path.format(group) + '/query')) != len(os.listdir(config.sfm_img_path.format(group) + '/query')):
                    generate_masks(config.sfm_img_path.format(group), config.sfm_mask_path.format(group))
                if not os.path.exists(os.path.join(config.sfm_sparse_path.format(group), 'points3D.txt')):
                    sfm(config.sfm_img_path.format(group), config.sfm_mask_path.format(group),  config.sfm_sparse_path.format(group), config.sfm_offline_path.format(group))
                
                do your verifications here (visibility, proper localization, detections outside of the intersection)
                if localization_check and not os.path.exists(os.path.join(config.sfm_sparse_path.format(group), 'localized_imgs.pkl')):
                    percent = get_well_localized_imgs(config.sfm_img_path.format(group), config.sfm_sparse_path.format(group))
                else:
                    percent = -1
                    
                get_change_gt(os.path.join(config.sfm_sparse_experiment.format(group, experiment)), group, ref_polygon, pretend_empty_map, query_time = datetime.strptime(date, '%Y_%m_%d').strftime("%m/%d/%y"))
                #this will change the label based on observability
                obs = get_observability(config.sfm_img_path.format(group), config.sfm_mask_path.format(group), config.sfm_sparse_path.format(group), config.sfm_offline_path.format(group), localization_check, experiment = experiment)
                
                if (localization_check and not os.path.exists(os.path.join(config.sfm_sparse_path.format(group), 'localization_query_det.pkl'))) or (not localization_check and not os.path.exists(os.path.join(config.sfm_sparse_path.format(group), 'query_det.pkl'))):
                    get_det(config.sfm_sparse_path.format(group), config.sfm_sparse_path.format(group), config.sfm_img_path.format(group), config.model_path, ref = False, debug = True, localization_check = localization_check)
                
                get_eval(config.sfm_sparse_path.format(group), experiment, localization_check)
                plt.close('all')
                m.update(os.path.join(config.sfm_sparse_experiment.format(group, experiment), 'change.pkl'))

                print('succeed {} {} {}'.format(group, date, experiment))    
            except:
                failed.append([group, date])
                print('failed {} {} {}'.format(group, date, experiment))
    result = m.show()
    tracker.update(x, result)
tracker.show()