#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:38:20 2022

@author: Tom Bu
"""
from utils.dynamic_obj_mask import generate_masks
from utils.automate_registration import sfm
import os
import pytz
import pickle

from utils.offline_detections import get_det
from utils.offline_query_eval import get_eval

from datetime import datetime, timezone
import shutil
import pickle

class metrics:
    def __init__(self):
        self.metrics = {'any_change_existing_cw':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                        'any_change_cw':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                        'any_change_intersection':{'tp':0, 'fp': 0, 'tn': 0, 'fn':0},
                        }

    def update(self, path):
        with open(path, 'rb') as f:
            new_data = pickle.load(f)          
        for k1, v1 in self.metrics.items():
            for k2, v2 in v1.items():
                self.metrics[k1][k2] += new_data[k1][k2]

    def show(self, silent = True):
        output = {}
        for key in self.metrics.keys():
            output[key] = {}
            precision = self.change_precision(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])
            recall = self.recall(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])
            no_change_precision = self.no_change_precision(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])
            accuracy = self.accuracy(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])
            fpr = self.fpr(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])
            f1 = self.f1(self.metrics[key]['tp'], self.metrics[key]['fp'], self.metrics[key]['tn'], self.metrics[key]['fn'])

            for x, y in zip(['precision', 'recall', 'no change precision', 'accuracy', 'fpr', 'f1'], [precision,recall, no_change_precision, accuracy, fpr, f1]):
                if not silent:
                    print(key, x, y)
                output[key][x] = y
        return output
    
    def change_precision(self, tp, fp, tn, fn):
        try:
            return tp/(tp+fp)
        except:
            return None
        
    def no_change_precision(self, tp, fp, tn, fn):
        try:
            return tn/(tn+fn)
        except:
            return None
    
    def accuracy(self, tp, fp, tn, fn):
        try:
            return (tp + tn)/(tp + tn + fp + fn)
        except:
            return None
    
    def recall(self, tp, fp, tn, fn):
        try:
            return tp/(tp+fn)
        except:
            return None
        
    def fpr(self, tp, fp, tn, fn):
        try:
            return fp / (tn + fp)
        except:
            return None
    
    def tpr(self, tp, fp, tn, fn):
        try:
            return tp/(tp + fn)
        except:
            return None
        
    def f1(self, tp, fp, tn, fn):
        precision = self.change_precision(tp, fp, tn, fn)
        recall = self.recall(tp, fp, tn, fn)
        try:
            return 2 * precision * recall / (precision + recall)
        except:
            return None

