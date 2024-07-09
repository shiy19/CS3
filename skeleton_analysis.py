import math
import numpy as np
from scipy.spatial.distance import cdist,pdist, squareform
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from skimage.util import invert
from skimage.graph import route_through_array
from scipy.spatial import distance
import copy
from skimage import io, color, measure, morphology, filters
import sys
from scipy.spatial import ConvexHull
from tqdm import tqdm
from skimage.measure import label, regionprops,LineModelND, ransac
from itertools import combinations
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter,polygon
from scipy.ndimage import convolve, distance_transform_edt
from skimage import io, color, filters, morphology
import json
import torch
import glob
from image_processing import *
from utils import *
from mask_processing import *


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)

def standardize_format(sperm_tail):
    standardized_data = []
    for item in sperm_tail:
        if isinstance(item, dict):
            standardized_item ={'segmentation': item['segmentation']}
        else:
            standardized_item = {'segmentation': item}
        standardized_data.append(standardized_item)
    return standardized_data

def filter_masks(masks, threshold=50):
    filtered_masks = []
    for mask_dict in masks:
        segmentation = mask_dict['segmentation']
        if np.sum(segmentation) >= threshold:
            filtered_masks.append(mask_dict)
    return filtered_masks

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def calculate_max_inscribed_distance(mask):
    y_coords, x_coords = np.where(mask)
    points = np.column_stack((x_coords, y_coords))
    if len(points) > 1:
        return max(pdist(points))
    else:
        return 0

def is_touching_border(mask):
    return np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])

def process_masks(data):
    masks_to_remove = []
    for i, item in enumerate(data):
        segmentation_mask = item['segmentation']
        max_distance = calculate_max_inscribed_distance(segmentation_mask)
        touches_border = is_touching_border(segmentation_mask)
        if max_distance < 30 and touches_border:
            masks_to_remove.append(i)
        elif max_distance <20:
            masks_to_remove.append(i)
    return masks_to_remove

def label_connected_components(layer):
    return label(layer, return_num=True)

def split_layer_into_components(layer):
    labeled, num_components = label_connected_components(layer)
    component_layers = []
    for component in range(1, num_components + 1):
        component_layer = (labeled == component)
        component_layers.append(component_layer)
    return component_layers

def find_and_draw_longest_line(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    longest_line = None
    max_distance = 0
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for i in range(len(approx)):
            for j in range(i + 1, len(approx)):
                distance = np.linalg.norm(approx[i][0] - approx[j][0])
                if distance > max_distance:
                    max_distance = distance
                    longest_line = (tuple(approx[i][0]), tuple(approx[j][0]))
    canvas = np.zeros_like(mask)
    if longest_line is not None:
        cv2.line(canvas, longest_line[0], longest_line[1], 255, 1)
    return canvas, longest_line

def remove_masks_with_border_objects(masks, threshold=15):
    filtered_masks = []
    for mask in masks:
        top_edge_count = np.sum(mask[0, :] > 0)
        bottom_edge_count = np.sum(mask[-1, :] > 0)
        left_edge_count = np.sum(mask[:, 0] > 0)
        right_edge_count = np.sum(mask[:, -1] > 0)
        total_edge_count = top_edge_count + bottom_edge_count + left_edge_count + right_edge_count
        if total_edge_count <= threshold:
            filtered_masks.append(mask)
    return np.array(filtered_masks)

def find_longest_line(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    longest_line = None
    max_distance = 0
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for i in range(len(approx)):
            for j in range(i + 1, len(approx)):
                distance = np.linalg.norm(approx[i][0] - approx[j][0])
                if distance > max_distance:
                    max_distance = distance
                    longest_line = (tuple(approx[i][0]), tuple(approx[j][0]))
    return longest_line

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >= 13)

def nearest_branch_point(endpoint, branch_points):
    y, x = np.where(branch_points)
    if len(x) == 0 or len(y) == 0:
        return endpoint, float('inf')
    distances = np.sqrt((x - endpoint[0])**2 + (y - endpoint[1])**2)
    nearest_index = np.argmin(distances)
    return (x[nearest_index], y[nearest_index]), distances[nearest_index]

def find_modified_endpoints(skeletons):
    all_endpoints = []
    for index, skeleton in enumerate(skeletons):
        branch_points = find_branch_points(skeleton)
        labeled_skeleton= label(skeleton)
        properties = regionprops(labeled_skeleton)
        endpoints = []
        for prop in properties:
            if prop.extent < 0.5:
                coords = prop.coords
                for coord in coords:
                    if np.sum(skeleton[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2]) == 2:
                        nearest_bp, distance = nearest_branch_point((coord[1], coord[0]), branch_points)
                        if distance < 25:
                            endpoints.append(nearest_bp)
                        else:
                            endpoints.append((coord[1], coord[0]))
        endpoints = list(set(endpoints))
        all_endpoints.append({'layer': index, 'endpoints': endpoints})
    return all_endpoints
