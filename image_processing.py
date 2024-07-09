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
from skeleton_analysis import *
from utils import *
from mask_processing import *


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))

def resize_image(input_path, output_path, width, height):
    if os.path.exists(input_path):
        img = Image.open(input_path)
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)
    else:
        print(f"File {input_path} does not exist.")

def adjust_brightness(image_path, target_brightness):
    with Image.open(image_path) as img:
        hsv_image = img.convert('HSV')
        h, s, v = hsv_image.split()
        v_array = np.array(v, dtype=np.float64)
        current_brightness = np.mean(v_array)
        if current_brightness == 0 or current_brightness == target_brightness:
            return img
        v_array *= target_brightness / current_brightness
        v_array[v_array > 255] = 255
        v_array = v_array.astype(np.uint8)
        adjusted_hsv_image = Image.merge('HSV', (h, s, Image.fromarray(v_array)))
        return adjusted_hsv_image.convert('RGB')

def adjust_rgb_channels(image_path, target_r, target_g, target_b):
    with Image.open(image_path) as img:
        r, g, b = img.split()
        r_avg, g_avg, b_avg = map(np.mean, (r, g, b))
        r_scale = target_r / r_avg if r_avg > 0 else 0
        g_scale = target_g / g_avg if g_avg > 0 else 0
        b_scale = target_b / b_avg if b_avg > 0 else 0
        r = (np.array(r) * r_scale).clip(0, 255).astype(np.uint8)
        g = (np.array(g) * g_scale).clip(0, 255).astype(np.uint8)
        b = (np.array(b) * b_scale).clip(0, 255).astype(np.uint8)
        adjusted_image = Image.merge('RGB', (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
        return adjusted_image

def dehaze_image(img, clip_limit=150, tile_size=(2, 2)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def sharpen_image(img, factor=1.235):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel * factor)

def adjust_contrast(img, factor=1.0):
    mean = np.mean(img)
    return np.clip((1 + factor) * (img - mean) + mean, 0, 255).astype(np.uint8)

def process_image_skimage(image_opencv):
    image_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    gray_sk = color.rgb2gray(image_rgb)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=2000)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_rgb[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_rgb > [200, 200, 200], axis=-1)
    image_rgb[whitening_mask_sk] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def process_single_image(image_path, news_path):
    image_opencv = cv2.imread(image_path)
    if image_opencv is None:
        print("can't load image")
        return
    img_original_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    img_new_adjustment = img_original_rgb.copy()
    img_new_adjustment = adjust_contrast(img_new_adjustment, 1.0)
    img_new_adjustment = sharpen_image(img_new_adjustment, 1.235)
    img_new_adjustment = dehaze_image(img_new_adjustment)
    img_noise_reduced = cv2.fastNlMeansDenoisingColored(img_new_adjustment, None, 20, 20, 7, 21)
    img_final = process_image_skimage(img_noise_reduced)
    cv2.imwrite(news_path, img_final)

def find_furthest_distance_in_each_layer(masks):
    distances = []
    for layer in masks:
        points = np.column_stack(np.where(layer > 0))
        if len(points) > 1:
            pairwise_distances = squareform(pdist(points, 'euclidean'))
            max_distance = pairwise_distances.max()
        else:
            max_distance = 0
        distances.append(max_distance)
    return distances

def calculate_furthest_distance(mask):
    y_coords, x_coords = np.nonzero(mask)
    points = np.column_stack((x_coords, y_coords))
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    distances = cdist(hull_points, hull_points, metric='euclidean')
    furthest_distance = np.max(distances)
    return furthest_distance

def remove_small_components(mask, min_size):
    labeled_array = label(mask)
    sizes = np.bincount(labeled_array.ravel())
    small_components = np.where(sizes < min_size)[0]
    remove_mask = np.in1d(labeled_array, small_components).reshape(labeled_array.shape)
    mask[remove_mask] = False
    return mask

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >= 13)

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, np.isin(convolved, [11, 21]))

def is_fully_connected(skeleton):
    labeled_skeleton, num_components = label(skeleton, return_num=True)
    return num_components == 1

def is_adjacent_to_branch(endpoint, branch_points):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor = (endpoint[0] + dx, endpoint[1] + dy)
            if neighbor in branch_points:
                return True
    return False

def process_skeletons(data):
    modified_skeletons = []
    for item in data:
        if 'segmentation' in item:
            segmentation_mask = item['segmentation']
            segmentation_mask_contiguous = np.ascontiguousarray(segmentation_mask, dtype=bool)
            skeleton = skeletonize(segmentation_mask_contiguous)
            branch_points = find_branch_points(skeleton)
            branch_point_coordinates = np.argwhere(branch_points)
            skeleton_without_branches = skeleton.copy()
            for coord in branch_point_coordinates:
                skeleton_without_branches[tuple(coord)] = 0
            labeled_skeleton, _ = label(skeleton_without_branches, return_num=True)
            small_components = [region for region in regionprops(labeled_skeleton) if region.area < 25]
            for component in small_components:
                component_mask = np.zeros_like(skeleton_without_branches, dtype=bool)
                for coord in component.coords:
                    component_mask[tuple(coord)] = True
                endpoints = find_endpoints(component_mask)
                endpoints_coordinates = np.argwhere(endpoints)
                if len(endpoints_coordinates) != 2:
                    continue
                branch_points_list = [tuple(coord) for coord in branch_point_coordinates]
                if not all(is_adjacent_to_branch(endpoint, branch_points_list) for endpoint in endpoints_coordinates):
                    for coord in component.coords:
                        skeleton_without_branches[tuple(coord)] = 0
            min_branch_points_needed = None
            min_num_branch_points = len(branch_point_coordinates) + 1
            if len(branch_point_coordinates) > 14:
                min_branch_points_needed = branch_point_coordinates
            else:
                for num_points_to_add in range(len(branch_point_coordinates) + 1):
                    for branch_subset in combinations(branch_point_coordinates, num_points_to_add):
                        temp_skeleton = skeleton_without_branches.copy()
                        for point in branch_subset:
                            temp_skeleton[tuple(point)] = 1
                        if is_fully_connected(temp_skeleton):
                            if num_points_to_add < min_num_branch_points:
                                min_num_branch_points = num_points_to_add
                                min_branch_points_needed = branch_subset
                            break
                    if min_branch_points_needed is not None:
                        break
            if min_branch_points_needed is not None:
                for point in min_branch_points_needed:
                    skeleton_without_branches[tuple(point)] = 1
            modified_skeletons.append(skeleton_without_branches)
    return np.array(modified_skeletons)

def analyze_masks_connectivity(masks_file):
    masks = masks_file
    disconnected_masks_indices = []
    connectcomponents = []
    for index, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        labeled_mask, num_components = label(mask, return_num=True, connectivity=2)
        connectcomponents.append(num_components)
        if num_components > 1:
            components_areas = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            coords = [np.column_stack(np.where(labeled_mask == i)) for i in range(1, num_components + 1)]
            min_distance = np.inf
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    distance = np.min(cdist(coords[i], coords[j]))
                    if distance < min_distance:
                        min_distance = distance
                        smallest_area = min(components_areas[i], components_areas[j])
            if min_distance > 25 or smallest_area < 50:
                disconnected_masks_indices.append(index)
    return disconnected_masks_indices, connectcomponents

def analyze_skeleton_connectivity(masks):
    disconnected_masks_indices = []
    for index, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        skeleton = skeletonize(np.ascontiguousarray(mask, dtype=bool))
        labeled_skeleton, num_components = label(skeleton, return_num=True)
        if num_components > 1:
            disconnected_masks_indices.append(index)
    return disconnected_masks_indices