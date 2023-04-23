#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:49:25 2022

@author: Tom Bu
"""

from utils.utils import read_images, read_cameras, get_extrinsic, project_3d_points, get_intrinsic
import numpy as np
from collections import OrderedDict
import os 
import pickle
import cv2
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon 
from matplotlib.patches import Patch
import descartes
from utils.offline_detections import get_3d_points, find_homography, transform_points, iou
from scipy.spatial import transform
import matplotlib.pyplot as plt

class dated_config:
    def __init__(self, day = None):
        self.model_path = "./model/crosswalk_detector/model_final.pth"
        self.sfm_path = './sfm/2022_10_16_offline_map_query/'
        self.sfm_query_path = self.sfm_path + '{}/query/' + day + '/'
        self.sfm_img_path = self.sfm_query_path + 'images'
        self.sfm_mask_path = self.sfm_query_path + 'mask'
        self.sfm_sparse_path = self.sfm_query_path + 'sparse'
        self.sfm_offline_path = self.sfm_path + '{}/sparse/plane_segmentation'

def read_3D(path):
    """
    input: path to root folder
    return: dictionary that maps 3D point ID to it's X, Y, Z value
    """
    pID2XYZ = OrderedDict()
    pID2all = OrderedDict()

    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    #TRACK[] is a list of images and their points which contain the 3D point
    points_path  = os.path.join(path, 'points3D.txt')

    with open(points_path) as f:
        lines = f.read().splitlines()
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else: 
            fields = line.split(' ')
            points_attr = np.array([float(pt) for pt in fields])
            #we discard the tracks, and the RGB value
            pID2XYZ[points_attr[0]] = np.array([points_attr[1],points_attr[2],points_attr[3]])
            pID2all[int(points_attr[0])] = points_attr[1:]
    return pID2XYZ, pID2all

def cam2world_position(q, t):
    """
    camera to world position 

    Parameters
    ----------
    q : np.array
        DESCRIPTION.
    t : np.array
        DESCRIPTION.

    Returns
    -------
    position : TYPE
        DESCRIPTION.

    """
    
    #make scalar last
    q = q[[1, 2, 3, 0]]
    #get rotation matrix from the quaternion. world to cam
    R_w2c = transform.Rotation.from_quat(q)
    
    # world to cam translation    
    position = - R_w2c.as_matrix().T.dot(t) 

    
    return position    


def get_observability(img, mask, query_sparse, offline_sparse, localization_check=False, experiment = ''): 
    """
    generate the crosswalks that can be observed by the query images (i.e. are they visible in the images according to the field of view and are not occluded)
    
    project into image and transform out of image
    account for occlusion and whether it appears in the frame
    generate a new label

    :param img: path to the query images
    :param mask: path to the query image masks
    :param query_sparse: path to the sparse reconstruction with the query image
    :param offline_sparse: path to the sparse reconstruction with the query image
    :param localization_check: boolean value to consider images that are localized well
    """
    
    if experiment:
        with open(os.path.join(query_sparse, experiment, 'label.pkl'), 'rb') as f:
            ref_detection = pickle.load(f)
        if 'obs' not in experiment:
            return {'observed':-1, 'total':len(ref_detection)}
    else:
        with open(os.path.join(offline_sparse, 'label.pkl'), 'rb') as f:
            ref_detection = pickle.load(f)        
    if localization_check:
        with open(os.path.join(query_sparse, 'localized_imgs.pkl'), 'rb') as f:
            well_localized_imgs = pickle.load(f)        
   
    img2pose, img2id, imgto2dfeatures = read_images(query_sparse, subdir = '', file = 'images.txt')
    pID2XYZ, pID2all = read_3D(query_sparse)
    cameras = read_cameras(query_sparse, subdir = '')
    points = get_3d_points(query_sparse)
    
   
    obs_detection = []
    obs_cnt = 0
    for gt_cw in ref_detection:
        gt_cw_id = gt_cw['id']
        gt_cw_bev = gt_cw['poly']
        
        x = MultiPolygon(gt_cw_bev).centroid.x
        y = MultiPolygon(gt_cw_bev).centroid.y
        gt_cw_center = np.array([x, y])    
        
        img_li = sorted(os.listdir(img + '/query'))
        cnt_close = 0
        cnt_full = 0
        cnt_not_seen = 0

        for i in img_li:
            if localization_check and i not in well_localized_imgs:
                cnt_not_seen += 1
                continue                  
            if i not in imgto2dfeatures:
                cnt_not_seen += 1
                continue        
            features = imgto2dfeatures[i]
            img_id = img2id[i]
            features = features[features[:, 2] != -1]
            
            [quaternion, translation, image_id, camera_id] = img2pose[i]
            pose = np.concatenate([quaternion, translation])
            extrinsic = get_extrinsic(pose)
            intrinsic = cameras[cameras[:, 8] == camera_id, :].flatten()
            
            #check the distance from the camera
            cam_position = cam2world_position(quaternion, translation)
            l2_dist = np.linalg.norm(cam_position[:2]-gt_cw_center)
            if l2_dist > 20:
                cnt_not_seen += 1
                continue
        
            
            #find homography transformation
            id3d_ref = points[:, 3:4]
            pixel_ref = points[:, 0:2]
            pixel_curr, id3d_curr = project_3d_points(points, extrinsic, intrinsic)
            M = find_homography(id3d_ref, pixel_ref, id3d_curr, pixel_curr)
        
            #if not ground points are in the image and no homography transformation can be found, then skip
            if M is None:
                cnt_not_seen += 1
                continue
        
            #project gt bev cw into the image plane. project to see if there's anything that projects into the image plane from in front of the camera. the homography will not be able to do this correctly
            gt_cw_img = []
            for gt_poly in gt_cw_bev:
                gt_points = np.array(gt_poly.exterior.coords[:-1])
                #append the height, which is 0
                gt_points = np.concatenate([gt_points, np.zeros((gt_points.shape[0], 1)), np.array([[x] for x in range(len(gt_points))])], axis = 1)
                
                pixel_curr = project_3d_points(gt_points, extrinsic, intrinsic, force_pinhole = True, img_plane = True)
                pixel_curr = pixel_curr[:, :2]
                
                if len(pixel_curr) >3:
                    gt_img_poly = Polygon(pixel_curr.tolist()) 
                    gt_cw_img.append(gt_img_poly)
            
        
            #see if there's anything
            if len(gt_cw_img) == 0:
                cnt_not_seen += 1
                continue
        
            #bev to image transformation using homography transformation. This will match the gt bev polygon correctly when you reverse it. the projection will be slightly off due to distortions
            gt_cw_img = transform_points(gt_cw_bev, np.linalg.inv(M))
            
            
            gt_cw_img = MultiPolygon(gt_cw_img)
            
            #get mask, turn mask into contours
            obj_mask = cv2.imread(mask + '/query/' + i + '.png')
            obj_mask = np.ones_like(obj_mask) * 255 - obj_mask
            
            #what's observed in the image
            h, w, c = obj_mask.shape
            obs_poly = Polygon([[0, 0], [w, 0], [w, h], [0, h]])
            
            #remove observability due to object
            obj_polygons = []
            if obj_mask.max() > 0:
                obj_contours, hierarchy= cv2.findContours((obj_mask[:, :, 0]>1).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # #remove any degenerate polygons from the contours like if two edges are aligned
                # obj_contours = [cv2.approxPolyDP(cnt, 1, True) for cnt in obj_contours]
                for obj_contour in obj_contours:
                    obj_contour = obj_contour[:, 0, :]
                    cntr_points = obj_contour.tolist()
                    while len(cntr_points) > 3:
        
                    # if len(cntr_points) > 3:
                        poly = Polygon(cntr_points)
                        if not poly.is_valid:
                            cntr_points = cntr_points[::5]
                            continue
                        else:
                            #subtract each object contour from the observable mask/polygon
                            obs_poly = obs_poly.difference(poly)
                            break
            
            #observable crosswalk in the image = the intersection of the ground truth crosswalk that's projected into the image and the observability mask
            obs_cw_img = gt_cw_img.intersection(obs_poly)
            if obs_cw_img.area == 0:
                cnt_not_seen += 1
                # print('cant see')
                continue
            if isinstance(obs_cw_img, Polygon):
                obs_cw_bev = transform_points([obs_cw_img], M)
            else:  
                obs_cw_bev = transform_points(list(obs_cw_img), M)
                
            iou_obs = iou(MultiPolygon(gt_cw_bev), MultiPolygon(obs_cw_bev))
            
            #due to rounding errors
            if iou_obs > .95:
                # print('fully seen')
                cnt_full += 1
            elif iou_obs > 0:
                # print('partially')
                cnt_close += 1
            else:
                # print('cant see')
                cnt_not_seen += 1
        # print('fully seen :', cnt_full)
        # print('partially seen :', cnt_close)
        # print('not seen :', cnt_not_seen)
    
        #indicate whether it was observed or not
        if cnt_full > 3:
            gt_cw['observed'] = True
            obs_cnt += 1
        else:
            gt_cw['observed'] = False
        obs_detection.append(gt_cw)
    if experiment:
        with open(os.path.join(query_sparse, experiment, 'label.pkl'), "wb") as poly_file:
            pickle.dump(obs_detection, poly_file, pickle.HIGHEST_PROTOCOL)        
    else:
        if localization_check:
            with open(os.path.join(query_sparse, 'localization_observed_label.pkl'), "wb") as poly_file:
                pickle.dump(obs_detection, poly_file, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(query_sparse, 'observed_label.pkl'), "wb") as poly_file:
                pickle.dump(obs_detection, poly_file, pickle.HIGHEST_PROTOCOL)
                
    fig = plt.figure(figsize=(9, 16))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.scatter(points[:, 0], points[:, 1], s = 0.01)
    for poly in obs_detection:
        if poly['observed']:
            ax.add_patch(descartes.PolygonPatch(MultiPolygon(poly['poly']), fc = 'b'))
        else:
            ax.add_patch(descartes.PolygonPatch(MultiPolygon(poly['poly']), fc = 'r'))
    b = Patch(facecolor='b', label='fully seen')
    r = Patch(facecolor='r', label='not fully seen')
    ax.legend(handles=[r, b])
    # plt.show()
    if experiment:
        plt.savefig(os.path.join(query_sparse, experiment, 'reference.jpg'))
    else:
        if localization_check:
            plt.savefig(os.path.join(query_sparse, 'localization_observed_reference.jpg'))
        else:
            plt.savefig(os.path.join(query_sparse, 'observed_reference.jpg'))
    plt.close(fig)
    return {'observed':obs_cnt, 'total':len(ref_detection)}

