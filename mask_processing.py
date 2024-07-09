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
from skeleton_analysis import *


def detect_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    endpoints = ((neighbors - 10) == 1) & skeleton
    return endpoints

def detect_branch_points(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    branch_points = ((neighbors - 10) >= 3) & skeleton
    return branch_points

def endpoints_to_nearest_branch_distance(skeleton, endpoints, branch_points):
    endpoints_coords = np.column_stack(np.where(endpoints))
    branch_points_coords = np.column_stack(np.where(branch_points))
    tree = cKDTree(branch_points_coords)
    distances, _ = tree.query(endpoints_coords, k=1)
    return distances

def find_point_on_skeleton(skeleton, start, distance):
    current_point = start
    visited = set()
    height, width = skeleton.shape
    for _ in range(abs(distance)):
        visited.add(current_point)
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                     for y2 in range(y - 1, y + 2)
                     if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) != current_point and skeleton[y2, x2] and (
                     x2, y2) not in visited]
        if not neighbors:
            break  
        current_point = neighbors[0] if distance > 0 else neighbors[-1]
    return current_point

def calculate_angle(point_a, point_b, point_c):
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.isin(filtered, [11, 21])

def is_near_endpoint(point, endpoints, max_distance=30):
    distance_map = distance_transform_edt(~endpoints)
    y, x = point
    return distance_map[y, x] <= max_distance

def find_sharp_angle_points(skeleton, distance=10, angle_threshold=130, max_endpoint_distance=30):
    sharp_points = []
    endpoints = find_endpoints(skeleton)
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] and not is_near_endpoint((y, x), endpoints, max_endpoint_distance):
                point_a = find_point_on_skeleton(skeleton, (x, y), distance)
                point_b = find_point_on_skeleton(skeleton, (x, y), -distance)
                if point_a != (x, y) and point_b != (x, y):
                    angle = calculate_angle(point_a, (x, y), point_b)
                    if angle < angle_threshold:
                        sharp_points.append((y, x))
    return sharp_points

def mark_points_on_skeleton(skeleton, points):
    marked_skeleton = gray2rgb(skeleton.astype(np.uint8) * 255)
    for point in points:
        y, x = point
        rr, cc = circle_perimeter(y, x, 1)
        marked_skeleton[rr, cc] = [255, 0, 0]
    return marked_skeleton

def find_closest_points(coords1, coords2):
    distances = cdist(coords1, coords2)
    idx_min = np.argmin(distances)
    min_idx_1, min_idx_2 = np.unravel_index(idx_min, distances.shape)
    return coords1[min_idx_1], coords2[min_idx_2]

def draw_filled_rectangle(mask, point_1, point_2, ratio):
    center_x = (point_1[1] + point_2[1]) / 2
    center_y = (point_1[0] + point_2[0]) / 2
    distance = np.linalg.norm(point_1 - point_2)
    dx = ratio / 2
    dy = distance / 2
    row_min = int(max(0, center_y - dy))
    row_max = int(min(mask.shape[0], center_y + dy))
    col_min = int(max(0, center_x - dx))
    col_max = int(min(mask.shape[1], center_x + dx))
    mask[row_min:row_max, col_min:col_max] = 1

def extract_segments_from_image(masks_path, image_path):
    masks = np.load(masks_path, allow_pickle=True)
    image = Image.open(image_path)
    image_np = np.array(image)
    extracted_images_paths = []
    for i, mask_dict in enumerate(masks):
        segmentation = mask_dict['segmentation']
        new_image = Image.new("RGB", image.size, (255, 255, 255))
        new_image_np = np.array(new_image)
        new_image_np[segmentation == 1] = image_np[segmentation == 1]
        new_image_pil = Image.fromarray(new_image_np)
        new_image_path = f'extracted_image{i}_0.jpg'
        new_image_pil.save(new_image_path)
        extracted_images_paths.append(new_image_path)
    return extracted_images_paths

def extract_and_place_on_canvas(image_array, min_pixel_size=100):
    gray_image = color.rgb2gray(image_array)
    binary_image = gray_image < 0.5
    labeled_image = measure.label(binary_image, connectivity=2)
    region_props = measure.regionprops(labeled_image)
    output_images = []
    for region in region_props:
        if region.area >= min_pixel_size:
            minr, minc, maxr, maxc = region.bbox
            region_image = image_array[minr:maxr, minc:maxc]
            mask = labeled_image[minr:maxr, minc:maxc] == region.label
            canvas = np.ones(image_array.shape, dtype=np.uint8) * 255
            for i in range(3):
                canvas[minr:maxr, minc:maxc, i] = np.where(mask, region_image[:, :, i], 255)
            output_images.append(canvas)
    return output_images

def extract_components_and_update_masks(image_array, masks_path, min_pixel_size=100):
    existing_masks = np.load(masks_path, allow_pickle=True)
    gray_image = color.rgb2gray(image_array)
    binary_image = gray_image < 0.5
    labeled_image = measure.label(binary_image, connectivity=2)
    region_props = measure.regionprops(labeled_image)
    updated_masks = list(existing_masks)
    for idx, region in enumerate(region_props, start=2):
        if region.area >= min_pixel_size:
            mask = labeled_image == region.label
            mask_dict = {
                'segmentation': mask.astype(bool),
                'area': region.area,
                'bbox': region.bbox,
                'predicted_iou': None,
                'point_coords': None,
                'stability_score': None,
                'crop_box': None
            }
            updated_masks.append(mask_dict)
    return updated_masks

def extract_and_skeletonize(data):
    skeletons = []
    for item in data:
        if 'segmentation' in item:
            segmentation_mask = item['segmentation']
            segmentation_mask_contiguous = np.ascontiguousarray(segmentation_mask, dtype=bool)
            skeleton = skeletonize(segmentation_mask_contiguous)
            skeletons.append(skeleton)
    return np.array(skeletons)
