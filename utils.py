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
from skeleton_analysis import *
from mask_processing import *


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_farthest_points(points):
    max_distance = 0
    farthest_points = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = calculate_distance(points[i], points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_points = [points[i], points[j]]
    return farthest_points

def find_point_on_skeleton(skeleton, start, distance):
    current_point = start
    visited = set()
    height, width = skeleton.shape
    for _ in range(distance):
        visited.add(current_point)
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2) for y2 in range(y - 1, y + 2) if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) != current_point and skeleton[y2, x2] and (x2, y2) not in visited]
        if not neighbors:
            break
        current_point = neighbors[0]
        visited.add(current_point)
    return current_point

def extend_endpoints_on_skeletons(skeletons, all_endpoints, distance):
    all_endpoints_20 = []
    for item in all_endpoints:
        layer = item['layer']
        endpoints = item['endpoints']
        skeleton = skeletons[layer]
        extended_points = []
        for point in endpoints:
            extended_point = find_point_on_skeleton(skeleton, point, distance)
            extended_points.append(extended_point)
        all_endpoints_20.append({'layer': layer, 'endpoints': extended_points})
    return all_endpoints_20

def value_of_point(point, hsv_img_array):
    return hsv_img_array[point[1], point[0], 2]

def keep_corresponding_minimum_value_point(all_endpoints, all_endpoints_20, hsv_img_array):
    corresponding_minimum_value_points_per_layer = []
    for layer_endpoints, layer_endpoints_20 in zip(all_endpoints, all_endpoints_20):
        layer = layer_endpoints['layer']
        if len(layer_endpoints['endpoints']) != len(layer_endpoints_20['endpoints']):
            continue
        minimum_value_point = None
        minimum_value = float('inf')
        for point, point_20 in zip(layer_endpoints['endpoints'], layer_endpoints_20['endpoints']):
            current_value = value_of_point(point_20, hsv_img_array)
            if current_value < minimum_value:
                minimum_value = current_value
                minimum_value_point = point
        if minimum_value_point is not None:
            corresponding_minimum_value_points_per_layer.append({'layer': layer, 'endpoints': [minimum_value_point]})
    return corresponding_minimum_value_points_per_layer

def find_branch_length(skeleton, start_point, visited, max_length=50):
    length = 0
    current_point = start_point
    while length < max_length:
        length += 1
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2) for y2 in range(y - 1, y + 2) if 0 <= x2 < skeleton.shape[0] and 0 <= y2 < skeleton.shape[1] and skeleton[x2, y2] and (x2, y2) != current_point and (x2, y2) not in visited]
        if not neighbors:
            break
        current_point = neighbors[0]
        visited.add(current_point)
    return length

def find_longest_branch_point(skeleton, current_point, points):
    max_length = 0
    best_point = None
    visited = set(points)
    for neighbor in [(x, y) for x in range(current_point[0] - 1, current_point[0] + 2) for y in range(current_point[1] - 1, current_point[1] + 2) if 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and skeleton[x, y] and (x, y) != current_point and (x, y) not in points]:
        branch_length = find_branch_length(skeleton, neighbor, visited)
        if branch_length > max_length:
            max_length = branch_length
            best_point = neighbor
    return best_point

def find_skeleton_points(skeleton, start_point, num_points=30, max_branch_length=100):
    points = [start_point]
    for _ in range(num_points - 1):
        current_point = points[-1]
        next_point = find_longest_branch_point(skeleton, current_point, points)
        if next_point is None:
            break
        points.append(next_point)
    return points

def calculate_angle2(points):
    points_array = np.array(points)
    model, _ = ransac(points_array, LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    line_origin = model.params[0]
    line_direction = model.params[1]
    last_point = points_array[-1]
    projected_point = line_origin + np.dot((last_point - line_origin), line_direction) * line_direction
    first_point = points_array[0]
    projected_point2 = line_origin + np.dot((first_point - line_origin), line_direction) * line_direction
    direction_to_start = projected_point2 - projected_point
    angle = math.atan2(direction_to_start[0], direction_to_start[1]) * 180 / math.pi
    return angle if angle >= 0 else angle + 360

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_matching_tails(heads, tails, threshold):
    selected_tails_for_heads = {}
    for head in heads:
        head_layer = head['layer']
        selected_tails = []
        for tail in tails:
            tail_layer = tail['layer']
            for head_endpoint in head['endpoints']:
                for tail_endpoint in tail['endpoints']:
                    if distance(head_endpoint, tail_endpoint) <= threshold:
                        selected_tails.append(tail_layer)
                        break
                if tail_layer in selected_tails:
                    break
        selected_tails_for_heads[head_layer] = list(set(selected_tails))
    return selected_tails_for_heads

def angle_between_points(p1, p2, p3):
    a = (p2[0] - p1[0], p2[1] - p1[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    inner_product = a[0]*b[0] + a[1]*b[1]
    len_a = math.sqrt(a[0]**2 + a[1]**2)
    len_b = math.sqrt(b[0]**2 + b[1]**2)
    return math.acos(inner_product / (len_a * len_b))

def closest_points(head, tail):
    min_dist = float('inf')
    closest_pair = None
    for h_point in head:
        for t_point in tail:
            dist = distance(h_point, t_point)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (h_point, t_point)
    return closest_pair

def angle_difference(closest_tail_point, h_point, tail_layer, angles_with_20_points):
    angle_to_h_point = math.degrees(math.atan2(h_point[1] - closest_tail_point[1], h_point[0] - closest_tail_point[0]))
    if angle_to_h_point < 0:
        angle_to_h_point += 360
    angle_in_data = next(item for item in angles_with_20_points if item['layer'] == tail_layer)['angles'][0]
    angle_diff = abs(angle_to_h_point - angle_in_data)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return angle_diff

def create_rectangle_from_midpoints(pt1, pt2, width):
    direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    length = np.linalg.norm(direction)
    direction = direction / length
    perp_direction = np.array([-direction[1], direction[0]])
    half_width = width / 2
    corner1 = pt1 + perp_direction * half_width
    corner2 = pt1 - perp_direction * half_width
    corner3 = pt2 - perp_direction * half_width
    corner4 = pt2 + perp_direction * half_width
    return np.array([corner1, corner2, corner3, corner4])

def has_multiple_components(mask):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    return len(props) >= 2

def update_masks_with_rectangles(content):
    for key, mask in content.items():
        if has_multiple_components(mask):
            labeled_mask = label(mask)
            regions = regionprops(labeled_mask)
            min_dist = np.inf
            closest_points = None
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    region1, region2 = regions[i], regions[j]
                    for coord1 in region1.coords:
                        for coord2 in region2.coords:
                            dist = distance.euclidean(coord1, coord2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_points = (coord1, coord2)
            width = 15
            rectangle_coords = create_rectangle_from_midpoints(closest_points[0], closest_points[1], width)
            rr, cc = polygon(rectangle_coords[:, 0], rectangle_coords[:, 1], mask.shape)
            mask[rr, cc] = 1
            content[key] = mask
    return content

def mask_to_polygon_points(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).tolist()
        polygons.append(polygon)
    return polygons