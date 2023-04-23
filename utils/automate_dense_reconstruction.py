#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:57:42 2022

@author: Tom Bu
"""

import os
import shutil


def fix_plane_segmentation_points(sparse, plane):
    """
    the plane_segmentation reconstruction doesn't have the correct points3D since all of the tracks and image IDs are removed. It also has the plane points. we want to remove these for the dense reconstruction
    sparse: original sfm path
    plane: plane segmentation folder path

    """
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    #TRACK[] is a list of images and their points which contain the 3D point
    pID2XYZ = {}
    with open(os.path.join(plane, 'points3D.txt')) as f:
        lines = f.read().splitlines()
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else: 
            fields = line.split(' ')
            if int(fields[4]) == 0 and int(fields[5]) == 255 and int(fields[6]) == 0:
                continue        
            #we discard the tracks, and the RGB value
            pID2XYZ[fields[0]] = [fields[1][:19],fields[2][:19],fields[3][:19]]
    
    with open(os.path.join(sparse, 'points3D.txt')) as f:
        lines = f.read().splitlines()
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else: 
            fields = line.split(' ')
            xyz = pID2XYZ[fields[0]]
            fields[1:4] = xyz
            lines[i] = ' '.join(fields)
    
    with open(os.path.join(plane, 'points3D.txt') , 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in lines)   
        
        
def dense_reconstruction(working_dir):
    plane = os.path.join(working_dir, 'sparse/plane_segmentation')
    sparse_tmp = os.path.join(working_dir, 'sparse_tmp')
    sparse = os.path.join(working_dir, 'sparse')
    dense = os.path.join(working_dir, 'dense')
    img = os.path.join(working_dir, 'images')    
    
    os.makedirs(sparse_tmp, exist_ok=True)
    shutil.copy(os.path.join(plane, 'cameras.txt'), sparse_tmp)
    shutil.copy(os.path.join(plane, 'images.txt'), sparse_tmp)
    shutil.copy(os.path.join(plane, 'points3D.txt'), sparse_tmp)
    
    fix_plane_segmentation_points(sparse, sparse_tmp)
    

    cmd = 'colmap image_undistorter --image_path {} --input_path {} --output_path {}'.format(img, sparse_tmp, dense)
    os.system(cmd)
    
    cmd = 'colmap patch_match_stereo --workspace_path {}'.format(dense)
    os.system(cmd)
    
    cmd = 'colmap stereo_fusion --workspace_path {} --output_path {}'. format(dense, os.path.join(dense, 'fused.ply'))
    os.system(cmd)
    
    #save some space so remove some of the intermediate files
    shutil.rmtree(sparse_tmp) 
    shutil.rmtree(os.path.join(dense, 'images'))
    shutil.rmtree(os.path.join(dense, 'sparse'))
    shutil.rmtree(os.path.join(dense, 'stereo'))
    