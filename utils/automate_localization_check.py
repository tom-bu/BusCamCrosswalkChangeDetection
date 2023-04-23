#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:23:23 2022

@author: Tom Bu
"""

from utils.utils import read_images
import numpy as np
import os 
import pickle
from utils.automate_observability import cam2world_position, read_3D
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

def get_well_localized_imgs(img, query_sparse):
    """
    return query images that are within 5 meters of a reference have a height difference less than 0.2 meters
    :param img: path to query images
    :param query_sparse: path to query sparse folder
    """

    img2pose, img2id, imgto2dfeatures = read_images(query_sparse, subdir = '', file = 'images.txt')
    pID2XYZ, pID2all = read_3D(query_sparse)
    
    img_li = os.listdir(img + '/query')
    
    img2xyz = {}
    refxyz = []
    ref_cnt = 0
    ref_days = set()
    for key in img2pose.keys():
        [quaternion, translation, image_id, camera_id] = img2pose[key]
        cam_position = cam2world_position(quaternion, translation)
        if key in img_li:
            img2xyz[key] =  cam_position      
        else:
            refxyz.append(cam_position)
            ref_cnt += 1
            ref_days.add(datetime.fromtimestamp(int(key.split('_')[1])).strftime('%Y-%m-%d'))
        

    refxyz = np.vstack(refxyz)
    nbrs = NearestNeighbors(n_neighbors = 4)
    #exclude the z axis 
    nbrs.fit(refxyz[:, :2])
    distances = []
    heights = []
    well_localized_imgs = []
    for i in img_li:
        if i in img2xyz:
            cam_position = img2xyz[i]
            #distance is L2 norm
            distance, indexes = nbrs.kneighbors(cam_position[:2][None, :])
            height = refxyz[indexes.flatten()][:,2]
            distances.append(distance[0, 0])
            heights.append(abs(height[0] - cam_position[2]))
            well_localized_imgs.append(i)
    
    #check if the image is within 5 meters ofa reference image and if the height difference is within 0.2 meters
    well_localized_imgs = [well_localized_imgs[i] for i in range(len(well_localized_imgs)) if (distances[i] < 5 and heights[i] <0.2) ]
    with open(os.path.join(query_sparse, 'localized_imgs.pkl'), "wb") as poly_file:
        pickle.dump(well_localized_imgs, poly_file, pickle.HIGHEST_PROTOCOL)

    h = np.array(heights)
    d = np.array(distances)
    #percentage of the query images that are within 5 meters of a reference have a height difference less than 0.2 meters
    percent = sum(h[d<5] < 0.2)/len(h[d<5])
    return percent