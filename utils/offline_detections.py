#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:29:34 2021

@author: tombu
"""

import numpy as np
from scipy.spatial import transform
import os
import matplotlib.pyplot as plt
import cv2
from utils.utils import read_cameras, get_intrinsic, get_extrinsic, project_3d_points, filter_3d_points
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from shapely.geometry import Polygon
import descartes
import pickle
from matplotlib.patches import Patch


def init_detector(modelpath, thresh = 0.8):
    
    li = ['Bird', 'Ground Animal', 'Ambiguous Barrier', 'Concrete Block', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Road Median', 'Road Side', 'Lane Separator', 'Temporary Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Driveway', 'Parking', 'Parking Aisle', 'Pedestrian Area', 'Rail Track', 'Road', 'Road Shoulder', 'Service Lane', 'Sidewalk', 'Traffic Island', 'Bridge', 'Building', 'Garage', 'Tunnel', 'Person', 'Person Group', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Dashed Line', 'Lane Marking - Straight Line', 'Lane Marking - Zigzag Line', 'Lane Marking - Ambiguous', 'Lane Marking - Arrow (Left)', 'Lane Marking - Arrow (Other)', 'Lane Marking - Arrow (Right)', 'Lane Marking - Arrow (Split Left or Straight)', 'Lane Marking - Arrow (Split Right or Straight)', 'Lane Marking - Arrow (Straight)', 'Lane Marking - Crosswalk', 'Lane Marking - Give Way (Row)', 'Lane Marking - Give Way (Single)', 'Lane Marking - Hatched (Chevron)', 'Lane Marking - Hatched (Diagonal)', 'Lane Marking - Other', 'Lane Marking - Stop Line', 'Lane Marking - Symbol (Bicycle)', 'Lane Marking - Symbol (Other)', 'Lane Marking - Text', 'Lane Marking (only) - Dashed Line', 'Lane Marking (only) - Crosswalk', 'Lane Marking (only) - Other', 'Lane Marking (only) - Test', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Parking Meter', 'Phone Booth', 'Pothole', 'Signage - Advertisement', 'Signage - Ambiguous', 'Signage - Back', 'Signage - Information', 'Signage - Other', 'Signage - Store', 'Street Light', 'Pole', 'Pole Group', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Cone', 'Traffic Light - General (Single)', 'Traffic Light - Pedestrians', 'Traffic Light - General (Upright)', 'Traffic Light - General (Horizontal)', 'Traffic Light - Cyclists', 'Traffic Light - Other', 'Traffic Sign - Ambiguous', 'Traffic Sign (Back)', 'Traffic Sign - Direction (Back)', 'Traffic Sign - Direction (Front)', 'Traffic Sign (Front)', 'Traffic Sign - Parking', 'Traffic Sign - Temporary (Back)', 'Traffic Sign - Temporary (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Vehicle Group', 'Wheeled Slow', 'Water Valve', 'Car Mount', 'Dynamic', 'Ego Vehicle', 'Ground', 'Static', 'Unlabeled']
    MetadataCatalog.get('mvd').set(thing_classes = li)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(li)
    cfg.MODEL.WEIGHTS = modelpath
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor


def get_detection(cam_path, predictor, args, show_img = True, show_polygon = False, save_img = False):
    im = cv2.imread(cam_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    idx_class = outputs["instances"].to("cpu").pred_classes == 45
    outputs["instances"] = outputs["instances"][idx_class]
    
    if len(outputs['instances']) != 0:
        predictions = outputs["instances"].to("cpu")
    
        
        polygons = []
        scores = []
        for (pred_mask, pred_score) in zip(predictions._fields['pred_masks'], predictions._fields['scores'].numpy()):
            pred_mask = pred_mask.numpy().astype('uint8')
            if pred_mask.max() > 0:
                contours, hierarchy= cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = np.concatenate(contours, axis = 0)
                hull= cv2.convexHull(cnt)
                hull = hull[:, 0, :]
                points = [(p0, p1) for (p0, p1) in zip(hull[:, 0], hull[:, 1])]
                if len(points) > 3:
                    poly = Polygon(points)
                    polygons.append(poly)
                    scores.append(pred_score)

    else:
        polygons = []
        scores = []
    if show_img:
        v = Visualizer(im[:, :, ::-1],
        metadata=MetadataCatalog.get('mvd'),
        scale=0.5,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(num = 2)
        plt.imshow(out.get_image())
    if save_img:
        v = Visualizer(im[:, :, ::-1],
        metadata=MetadataCatalog.get('mvd'),
        scale=1,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        parent = os.path.dirname(cam_path).split('/')[-2]
        os.makedirs(os.path.join(args.out_path, parent), exist_ok=True)    
        predictions = outputs["instances"].to("cpu")            
        cv2.imwrite(os.path.join(args.out_path, parent, os.path.basename(cam_path)), out.get_image()[:, :, ::-1])
        
    if show_polygon:
        fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.imshow(im[:, :, [2, 1, 0]])
        for poly in polygons:
            ax.add_patch(descartes.PolygonPatch(poly))

    return polygons, scores

def calc_iogt(gt, pred):
    """
    return intersection over the ground truth
    """    
    # Calculate Intersection and union, and tne IOU
    polygon_intersection = gt.intersection(pred).area
    gt_area = gt.area
    iogt = polygon_intersection / gt_area 
    return iogt

def get_3d_points(path):
    """
    Parameters
    ----------
    path : str
        DESCRIPTION.

    Returns
    -------
    points3d : float16
        x, y, z, 3d ID of 3d points in the world frame.

    """
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    #TRACK[] is a list of images and their points which contain the 3D point
    path  = os.path.join(path, 'points3D.txt')
    points3d = np.loadtxt(path, usecols = (1, 2, 3, 0, 4, 5, 6))
    points3d = points3d[(points3d[:, 4] == 0) & (points3d[:, 5] == 255) & (points3d[:, 6] == 0), :4]
    return points3d

def read_image_points(path):
    """
    input: root path
    
    returns
    ---------
    imgto2dfeatures
        dictionaries of image file names to the 2D features 
    """     
    imgto2dfeatures = {}
    path  = os.path.join(path, 'images.txt')
    with open(path) as f:
        lines = f.read().splitlines()

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else:
            if i % 2 == 0:
                fields = line.split(' ')
                #NAME
                image_name = os.path.basename(fields[-1])
                #IMAGE_ID
                image_id = int(fields[0])                
            else:
                fields = line.split(' ')
                points_2d = np.array([float(pt) for pt in fields])
                points_2d = np.reshape(points_2d, (-1, 3))
                #maps the name to 2d points in the image
                imgto2dfeatures[image_id] = points_2d
    return imgto2dfeatures

def read_images(path):
    """
    input: root path
    returns: images: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID
             names: list of image file names 
    """     
    path  = os.path.join(path, 'images.txt')
    images = np.loadtxt(path, usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8))
    images = images[::2]
    names = np.loadtxt(path, usecols = (9), dtype='str')
    names = names[::2]
    names = names.tolist()
    
    return images, names

def get_projections(name, names, images, cameras, points, force_pinhole = True, show_img = False, cam_path = None):
    '''
    get the points that are projected into the image. return the pixel coordinates in the image frame as well as the 3D point ID 

    Parameters
    ----------
    name : str
        name of the image.
    names : list
        list of all images.
    images : array
        image pose array.
    cameras : array
        camera intrinsics array.
    force_pinhole : TYPE, optional
        if you want to force a pinhole camera model. The default is True.

    Returns
    -------
    pixel_full : array (n x 3)
        pixel coordinates of 3D point projections.
    id3d_full : array (n x 1)
        3D point IDs

    '''
    i = names.index(name)
    image = images[i]
    image_id = int(image[0])
    
    pose = image[1:8]
    cameraID = int(image[8])
    
    intrinsic = cameras[cameras[:, 8] == cameraID, :].flatten()
    
    #camera parameters
    extrinsic = get_extrinsic(pose)
    pixel_full, id3d_full = project_3d_points(points, extrinsic, intrinsic, force_pinhole = force_pinhole)

    if show_img:
        assert cam_path is not None
        im = cv2.imread(cam_path)

        plt.figure(num = 2)
        plt.scatter(pixel_full[:, 0], pixel_full[:, 1])
        plt.imshow(im[:, :, [2, 1, 0]])

    return pixel_full, id3d_full

def get_position(name, names, images):
    '''
    get the location of the camera in x, y, z coordinates
    
    Parameters
    ----------
    name : str
        name of the image.
    names : list
        list of all images.
    images : array
        image pose array.
        
    Returns
    -------
    position : array (3 x 1)
        3D location of the camera
    '''
    i = names.index(name)
    image = images[i]
    image_id = int(image[0])
    pose = image[1:8]
    
    q = pose[:4]
    #make scalar last
    q = q[[1, 2, 3, 0]]
    #get rotation matrix from the quaternion. world to cam
    R_w2c = transform.Rotation.from_quat(q)
    
    # world to cam translation
    t= pose[4:]
    
    position = - R_w2c.as_matrix().T.dot(t) 

    
    return position

def find_homography(id3d_ref, pixel_ref, id3d_curr, pixel_curr):
    '''
    return the homography matrix that converts the ground plane of the current frame into the reference frame
    Parameters:
    pixel_ref : array (n x 3)
        pixel coordinates of 3D point projections.
    id3d_ref : array (n x 1)
        3D point IDs
    pixel_curr : array (n x 3)
        pixel coordinates of 3D point projections.
    id3d_curr : array (n x 1)
        3D point IDs
    '''
    _, idx_ref, idx_curr = np.intersect1d(id3d_ref.flatten(), id3d_curr.flatten(), return_indices=True)

    pixel_ref = pixel_ref[idx_ref, :]
    pixel_curr = pixel_curr[idx_curr, :] 
    if len(pixel_ref) >= 4 and len(pixel_curr) >= 4:
        M, status = cv2.findHomography(pixel_curr[:, :2], pixel_ref[:, :2], cv2.RANSAC)
        return M
    else:
        return None


def transform_points(curr_detection, M, cam_path = None, show_polygon = False):
    """
    transform points from one image to another image or from the image onto the ground 
    """

    if M is None:
        "failed homography"
        return []
    polygons = []
    for poly in curr_detection:
        hull = np.array(poly.exterior.coords[:-1])
        hull = np.concatenate((hull, np.ones((hull.shape[0], 1))), axis = 1)
    
        hull = hull.T
        hull = M @ hull
        hull = hull.T
        hull = hull/hull[:, 2:3]
    
        points = [(p0, p1) for (p0, p1) in zip(hull[:, 0], hull[:, 1])]
        poly = Polygon(points)
        if poly.is_valid:
            polygons.append(poly)
        else:
            continue
            # print('error in transformed polygon')
        
    if show_polygon and cam_path != None:
        im = cv2.imread(cam_path)
        fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.imshow(im[:, :, [2, 1, 0]])
        for poly in polygons:
            ax.add_patch(descartes.PolygonPatch(poly))
    return polygons
    
def sum_dict(total, single):
    for key, value in single.items():
        total[key] = value + total.get(key, 0)
    return total

def determine_valid_distance(position, center, coef, multiplier = 3):
    '''
    determine if a detection is close enough to the car to be considered an accurate detection

    Parameters
    ----------
    position : np.array (3,)
        position of the camera.
    center : np.array (3,)
        centroid of the detection. we assuem that the detection's height is 0.
    coef : np.array(4, 1)
        coefficients of the plane
    Returns
    -------
    TYPE: booleon
        is the detection within a certain multiple of the camera height.

    '''
    l2_dist = np.linalg.norm(position-center)
    # cam_height = position[2]
    cam_height = dist(position[np.newaxis, :], coef)
    return l2_dist < cam_height * multiplier

def dist(data, coef):
    A = np.ones((data.shape[0], 1))
    A = np.concatenate([data, A], axis = 1)
    d = np.abs(np.dot(A, coef)) / np.sqrt(np.sum((coef ** 2)[:3]))
    return d

def fit_plane(data):
    A = np.ones((data.shape[0], 1))
    A = np.concatenate([data, A], axis = 1)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    coef = vt[3, :]    
    return coef

def iou(poly1, poly2):
    """
    return intersection over union
    """
    polygon_intersection = poly1.intersection(poly2).area
    # polygon_union = poly1.union(poly2).area
    # polygon_union = min(poly1.area, poly2.area)
    polygon_union = poly1.area + poly2.area - polygon_intersection
    IOU = polygon_intersection / polygon_union 
    return IOU



def nms(total_poly, total_scores, thresh_iou=0.8):
    """
    perform nms on polygons

    Parameters
    ----------
    total_poly : list
        polygons.
    total_scores : list
        scores.
    thresh_iou : int, optional
        DESCRIPTION. The default is 0.8.

    Returns
    -------
    keep_poly : list
        polygons.
    keep_score : list
        scores.

    """
    P = total_poly

    # we extract the confidence scores as well
    scores = np.array(total_scores)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for 
    # filtered prediction boxes
    keep_poly = []
    keep_score = []
    keep_idx = []

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep_poly.append(P[idx])
        keep_score.append(total_scores[idx])
        keep_idx.append(idx)
        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        
        
        poly2 = P[idx]
        
        IoU = []
        for i in order:
            poly1 = P[i]   
            IoU.append(iou(poly1, poly2))
        IoU = np.array(IoU)

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
    
    return keep_poly, keep_score, keep_idx


def perform_multi_image_check(total_poly, total_scores, image_id, thresh_iou = 0.1, multi_image_threshold = 3):
    keep_poly = []
    keep_score = []
    index = []
    for i in range(len(total_poly)):
        seen_across_n_images = 1
        for j in range(len(total_poly)):
            if image_id[i] != image_id[j]:
                IoU = iou(total_poly[i], total_poly[j])
                if IoU > thresh_iou:
                    seen_across_n_images += 1
                    if seen_across_n_images >= multi_image_threshold:
                        keep_poly.append(total_poly[i])
                        keep_score.append(total_scores[i])
                        index.append(i)
                        break
    return keep_poly, keep_score, index

def query_detection(parent, args, check_valid_distance = False):
    
    points = get_3d_points(os.path.join(args.path, parent))
    points = points[::5, :]
    
    coef = fit_plane(points[:, :3])
    
    id3d_ref = points[:, 3:4]
    pixel_ref = points[:, 0:2]
    
    images, names = read_images(os.path.join(args.path, parent))
    cameras = read_cameras(os.path.join(args.path, parent), subdir = '')
    
    if args.localization_check:
        with open(os.path.join(args.path, 'localized_imgs.pkl'), 'rb') as f:
            well_localized_imgs = pickle.load(f)        
        
    logs = {}
    for i in names:
        date = i.split('/')[0]
        if date not in logs.keys():
            logs[date] = []
        logs[date].append(i)
    change_logs = logs

   
############### plot detection for all sequences ##################    
    if args.debug:
        track_det = []
    log_poly = {}
    #'2015_04_07', '2017_02_04', '2019_09_26', '2021_06_16'
    change_logs = ['query']
    for curr_log in change_logs:
        # curr_log = '2019_09_26'
        total_poly = []
        total_scores = []
        image_id = []
        for curr_img in logs[curr_log]:
    
            if args.localization_check:
                if curr_img.split('/')[1] not in well_localized_imgs:
                    continue
                            
            #detect crosswalk
            curr_path = os.path.join(args.img_path, curr_img)
            curr_detection, curr_scores = get_detection(cam_path = curr_path, predictor = args.predictor, show_img = False, save_img = args.save_pred, args = args)
            pixel_curr, id3d_curr = get_projections(curr_img, names, images, cameras, points, force_pinhole = True)
            # curr_detection, curr_scores = nms(curr_detection, curr_scores, thresh_iou = 0.1)
        
            M = find_homography(id3d_ref, pixel_ref, id3d_curr, pixel_curr)
            curr3d_detection = transform_points(curr_detection, M, None, False)
            
            #check detection distance from the vehicle
            
            position = get_position(curr_img, names, images)
    
            for i, (poly, score) in enumerate(zip(curr3d_detection, curr_scores)):

                x = poly.centroid.x
                y = poly.centroid.y
                z = (-coef[0] * x - coef[1] * y - coef[-1]) * 1. /coef[2]
                center = np.array([x, y, z])      
                if check_valid_distance:
                    if not determine_valid_distance(position, center, coef, multiplier= 20):
                        continue
                total_poly.append(poly)
                total_scores.append(score)
                image_id.append(curr_img)
                if args.debug:
                    track_det.append([curr_log, curr_img, curr_detection[i], curr3d_detection[i]])                    
        if args.multi_image:
            n = len(total_poly)
            total_poly, total_scores, index = perform_multi_image_check(total_poly, total_scores, image_id, thresh_iou = 0.1)
            if args.debug:
                track_det_p1 = track_det[:len(track_det) - n]
                track_det2 = track_det[len(track_det) - n:]
                track_det2 = [x for i, x in enumerate(track_det2) if i in index]
                track_det = track_det_p1 + track_det2
           

        #perform NMS
        total_poly_nms, total_scores_nms, keep_idx = nms(total_poly, total_scores, thresh_iou = 0.01)
        if args.debug:
            track_det_p1 = track_det[:len(track_det) - len(total_poly)]
            track_det2 = track_det[len(track_det) - len(total_poly):]
            track_det2 = [x for i, x in enumerate(track_det2) if i in keep_idx]
            track_det = track_det_p1 + track_det2
        log_poly[curr_log] = total_poly_nms
   
    ref_log = 'query'
    ref_detection = log_poly[ref_log] 
    if args.localization_check:
        with open(os.path.join(args.path, parent, 'localization_query_det.pkl'), "wb") as poly_file:
            pickle.dump(ref_detection, poly_file, pickle.HIGHEST_PROTOCOL)        
    else:
        with open(os.path.join(args.path, parent, 'query_det.pkl'), "wb") as poly_file:
            pickle.dump(ref_detection, poly_file, pickle.HIGHEST_PROTOCOL)

    fig = plt.figure(figsize=(9, 16))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title('{} to {}'.format(ref_log.split('_')[0], curr_log.split('_')[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.scatter(points[:, 0], points[:, 1], s = 0.01)
    for poly in ref_detection:
        ax.add_patch(descartes.PolygonPatch(poly, fc = 'b'))
    r = Patch(facecolor='b', label='reference')
    ax.legend(handles=[r])
    if args.localization_check:
        plt.savefig(os.path.join(args.path, parent, 'localization_query.jpg'))
    else:
        plt.savefig(os.path.join(args.path, parent, 'query.jpg'))



def ref_detection(parent, args, check_valid_distance = False):
    
    points = get_3d_points(os.path.join(args.path, parent))
    points = points[::5, :]
    
    coef = fit_plane(points[:, :3])
    
    id3d_ref = points[:, 3:4]
    pixel_ref = points[:, 0:2]
    
    images, names = read_images(os.path.join(args.path, parent))
    cameras = read_cameras(os.path.join(args.path, parent), subdir = '')
    
    
    logs = {}
    for i in names:
        date = i.split('/')[0]
        if date not in logs.keys():
            logs[date] = []
        logs[date].append(i)
    change_logs = logs

   
############### plot detection for all sequences ##################    
    if args.debug:
        track_det = []
    log_poly = {}
    #'2015_04_07', '2017_02_04', '2019_09_26', '2021_06_16'
    change_logs = ['camera3']
    for curr_log in change_logs:
        # curr_log = '2019_09_26'
        total_poly = []
        total_scores = []
        image_id = []
        for curr_img in logs[curr_log]:
            #detect crosswalk

            curr_path = os.path.join(args.img_path, curr_img)
            curr_detection, curr_scores = get_detection(cam_path = curr_path, predictor = args.predictor, show_img = False, save_img = args.save_pred, args = args)
            pixel_curr, id3d_curr = get_projections(curr_img, names, images, cameras, points, force_pinhole = True)
            # curr_detection, curr_scores = nms(curr_detection, curr_scores, thresh_iou = 0.1)
        
            M = find_homography(id3d_ref, pixel_ref, id3d_curr, pixel_curr)
            curr3d_detection = transform_points(curr_detection, M, None, False)
            
            #check detection distance from the vehicle
            
            position = get_position(curr_img, names, images)
    
            for i, (poly, score) in enumerate(zip(curr3d_detection, curr_scores)):

                x = poly.centroid.x
                y = poly.centroid.y
                z = (-coef[0] * x - coef[1] * y - coef[-1]) * 1. /coef[2]
                center = np.array([x, y, z])      
                if check_valid_distance:
                    if not determine_valid_distance(position, center, coef, multiplier= 20):
                        continue
                total_poly.append(poly)
                total_scores.append(score)
                image_id.append(curr_img)
                if args.debug:
                    track_det.append([curr_log, curr_img, curr_detection[i], curr3d_detection[i]])                    
        if args.multi_image:
            n = len(total_poly)
            total_poly, total_scores, index = perform_multi_image_check(total_poly, total_scores, image_id, thresh_iou = 0.1)
            if args.debug:
                track_det_p1 = track_det[:len(track_det) - n]
                track_det2 = track_det[len(track_det) - n:]
                track_det2 = [x for i, x in enumerate(track_det2) if i in index]
                track_det = track_det_p1 + track_det2
           

        #perform NMS
        total_poly_nms, total_scores_nms, keep_idx = nms(total_poly, total_scores, thresh_iou = 0.01)
        if args.debug:
            track_det_p1 = track_det[:len(track_det) - len(total_poly)]
            track_det2 = track_det[len(track_det) - len(total_poly):]
            track_det2 = [x for i, x in enumerate(track_det2) if i in keep_idx]
            track_det = track_det_p1 + track_det2
        log_poly[curr_log] = total_poly_nms
   
    ref_log = 'camera3'
    ref_detection = log_poly[ref_log] 
    with open(os.path.join(args.path, parent, 'ref_det.pkl'), "wb") as poly_file:
        pickle.dump(ref_detection, poly_file, pickle.HIGHEST_PROTOCOL)

    fig = plt.figure(figsize=(9, 16))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title('{} to {}'.format(ref_log.split('_')[0], curr_log.split('_')[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.scatter(points[:, 0], points[:, 1], s = 0.01)
    for poly in ref_detection:
        ax.add_patch(descartes.PolygonPatch(poly, fc = 'b'))
    r = Patch(facecolor='b', label='reference')
    ax.legend(handles=[r])
    plt.savefig(os.path.join(args.path, parent, 'reference.jpg'))


class Argument:
    def __init__(self, path, out_path, img_path, debug, save_pred, removal, localization_check):
        self.path = path
        self.out_path = out_path
        self.img_path = img_path
        self.debug = debug
        self.save_pred = save_pred
        self.removal = removal
        self.localization_check = localization_check

def get_det(path, out_path, img_path, modelpath, ref = True, debug = False, save_pred = True, multi_image = True, removal = True, threshold = 0.99, localization_check = False):
    '''
    :param path: path to your sparse reconstruction
    :param out_path: path to where to save images with detections
    :param img_path: path to images
    :param modelpath: path to model
    :param debug: DESCRIPTION, defaults to False
    :param save_pred: DESCRIPTION, defaults to True
    :param multi_image: perform multi-frame consistency check, defaults to True
    :param removal: DESCRIPTION, defaults to True
    :param threshold: model confidence threshold, defaults to 0.99
    '''
    args = Argument(path, out_path, img_path, debug, save_pred, removal, localization_check)
    predictor = init_detector(modelpath, thresh=threshold) 
    args.predictor = predictor
    args.multi_image = multi_image
    if ref:
        ref_detection('', args)
    else:
        query_detection('', args)
