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
from mask_processing import *
sys.path.append("segment-anything-main")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

I = "3392"
def set_I(new_value):
    global I
    I = new_value

Sperm_tail=[]


#The first step:Image Processing Workflow
def process_image(I):
    input_path = f"data/Preprocessing_images/{I}.jpg"  
    output_path = "data/original_images/new.jpg" 
    width = 1440         
    height = 1080       
    resize_image(input_path, output_path, width, height)
    target_brightness = 220
    image_path = 'data/original_images/new.jpg' 
    adjusted_image = adjust_brightness(image_path, target_brightness)
    new_path = 'data/original_images/new1.jpg'
    adjusted_image.save(new_path) 
    target_r = 216.5
    target_g = 212.5
    target_b = 219.5
    image_path = 'data/original_images/new1.jpg' 
    adjusted_image = adjust_rgb_channels(image_path, target_r, target_g, target_b)
    new_path = 'data/original_images/new2.jpg'
    adjusted_image.save(new_path) 
    image_path = 'data/original_images/new2.jpg'  
    news_path = f'data/Preprocessing_images/new2.jpg'
    process_single_image(image_path, news_path)
    input_path = f"data/Preprocessing_images/new2.jpg"
    output_path = f"data/Preprocessing_images/new2.jpg"
    width = 720         
    height = 540        
    resize_image(input_path, output_path, width, height)
    
process_image(I)


#The second step:Image Segmentation and Mask Processing
image = cv2.imread(f'data/Preprocessing_images/{I}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
model_type = "default"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
masks = np.array([mask for mask in masks if mask['area'] <= 100000])
np.save('masks2.npy', masks)
masks = np.load('masks2.npy', allow_pickle=True)
image_path = f'data/original_images/{I}.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
purple_hsv_min = np.array([100, 30, 20])
purple_hsv_max = np.array([180, 255, 255])
green_hsv_min = np.array([60, 0, 160])
green_hsv_max = np.array([130, 50, 230])
segmentation_masks = [mask['segmentation'] for mask in masks if 'segmentation' in mask]
valid_mask_indices_updated_800_400 = []
valid_masks = []
for index, segmentation_mask in enumerate(segmentation_masks):
    segmented_image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=segmentation_mask.astype(np.uint8))
    mask_purple = cv2.inRange(segmented_image_hsv, purple_hsv_min, purple_hsv_max)
    mask_green = cv2.inRange(segmented_image_hsv, green_hsv_min, green_hsv_max)
    num_purple_pixels = np.count_nonzero(mask_purple)
    num_green_pixels = np.count_nonzero(mask_green)
    if num_purple_pixels >= 800 and num_green_pixels >= 600:
        valid_mask_indices_updated_800_400.append(index)
        valid_masks.append(segmentation_mask) # Store the valid mask
np.save('valid_masks.npy', np.array(valid_masks))
image_path = f'data/Preprocessing_images/{I}.jpg'
image = Image.open(image_path)
masks_path = 'valid_masks.npy'
masks = np.load(masks_path)
image_info = (image.size, image.mode)
masks_info = masks.shape
image_np = np.array(image)
for mask in masks:
    image_np[mask == True] = [255, 255, 255]
modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image.jpg"
modified_image.save("modified_image.jpg")
image_sk = io.imread("modified_image.jpg")
gray_sk = color.rgb2gray(image_sk)
threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value
cleaned = morphology.remove_small_objects(binary, min_size=50)
mask = np.where(cleaned == 0, 1, 0).astype(bool)
image_sk[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]
processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)


#The third step:Iterative Mask Refinement and Image Whitening
T=True
XX=0
while T!=False:
    image = cv2.imread('modified_image_dilated.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = np.array([mask for mask in masks if mask['area'] <= 100000])
    np.save('masks2.npy', masks)
    masks = np.load('masks2.npy', allow_pickle=True)
    image_path = f'data/original_images/{I}.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    purple_hsv_min = np.array([100, 30, 20])
    purple_hsv_max = np.array([180, 255, 255])
    green_hsv_min = np.array([60, 0, 160])
    green_hsv_max = np.array([130, 50, 230])
    segmentation_masks = [mask['segmentation'] for mask in masks if 'segmentation' in mask]
    valid_mask_indices_updated_800_400 = []
    for index, segmentation_mask in enumerate(segmentation_masks):
        segmented_image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=segmentation_mask.astype(np.uint8))
        mask_purple = cv2.inRange(segmented_image_hsv, purple_hsv_min, purple_hsv_max)
        mask_green = cv2.inRange(segmented_image_hsv, green_hsv_min, green_hsv_max)
        num_purple_pixels = np.count_nonzero(mask_purple)
        num_green_pixels = np.count_nonzero(mask_green)
        if num_purple_pixels >= 800 and num_green_pixels >= 600:
            valid_mask_indices_updated_800_400.append(index)
            valid_masks.append(segmentation_mask)
    np.save('valid_masks.npy', np.array(valid_masks))
    if XX==len(valid_masks):
        T=False
        break
    XX=len(valid_masks)
    image_path = 'modified_image_dilated.jpg'
    image = Image.open(image_path)
    masks_path = 'valid_masks.npy'
    masks = np.load(masks_path)
    image_np = np.array(image)
    for mask in masks:
        image_np[mask == True] = [255, 255, 255]
    modified_image = Image.fromarray(image_np)
    processed_img_path = "modified_image.jpg"
    modified_image.save("modified_image.jpg")
    image_sk = io.imread("modified_image.jpg")
    gray_sk = color.rgb2gray(image_sk)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_sk[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]
    processed_img_path = "modified_image_dilated.jpg"
    cv2.imwrite(processed_img_path, image_sk)


#The fourth step:Mask Generation and Image Whitening  
image = cv2.imread('modified_image_dilated.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
masks = np.array([mask for mask in masks if mask['area'] <= 100000])
np.save('masks2.npy', masks)
masks1 = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
masks1 = np.stack(masks1)
image_path = f'data/original_images/{I}.jpg'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_color = np.array([100, 20, 20])
upper_color = np.array([180, 255, 255])
mask = cv2.inRange(hsv, lower_color, upper_color)
filtered_masks = []
masks222 = masks1

for current_mask in masks1:
    rows, cols = np.where(current_mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        continue
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    region_mask = mask[min_row:max_row+1, min_col:max_col+1]
    region_current_mask = current_mask[min_row:max_row+1, min_col:max_col+1]
    overlap = np.sum((region_mask == 255) & (region_current_mask == 1))
    total_pixels = np.sum(region_current_mask == 1)
    overlap_percentage = (overlap / total_pixels) * 100
    if overlap_percentage > 40:
        filtered_masks.append(current_mask)

filtered_masks = np.array(filtered_masks)
np.save('filtered_masks.npy', filtered_masks)

file_path = 'filtered_masks.npy'
masks = np.load(file_path)
furthest_distances = find_furthest_distance_in_each_layer(masks)
furthest_distances_array = np.array(furthest_distances)
masks_filtered = masks[furthest_distances_array <= 75]
np.save(file_path, masks_filtered)
file_path = 'filtered_masks.npy'
masks = np.load(file_path)
image_path = 'modified_image.jpg'
image = Image.open(image_path)
masks_path = 'filtered_masks.npy'
masks = np.load(masks_path)
image_np = np.array(image)

for mask in masks:
    image_np[mask == 1] = [255, 255, 255]

modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image.jpg"
modified_image.save("modified_image.jpg")
image_sk = io.imread("modified_image.jpg")
gray_sk = color.rgb2gray(image_sk)
threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value
cleaned = morphology.remove_small_objects(binary, min_size=50)
mask = np.where(cleaned == 0, 1, 0).astype(bool)
image_sk[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]
processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)

image22 = cv2.imread(r"modified_image_dilated.jpg")
image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)
mask_generator22 = SamAutomaticMaskGenerator(sam)
masks22 = mask_generator22.generate(image22)
masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
np.save("masks22.npy", masks22)
masks1 = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks22]
masks1 = np.stack(masks1)
image_path = f'data/original_images/{I}.jpg'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_color = np.array([100, 33, 20])
upper_color = np.array([180, 255, 255])
mask = cv2.inRange(hsv, lower_color, upper_color)
filtered_masks = list(np.load("filtered_masks.npy"))
masks222 = masks1
new = []

for current_mask in masks1:
    rows, cols = np.where(current_mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        continue
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    region_mask = mask[min_row:max_row+1, min_col:max_col+1]
    region_current_mask = current_mask[min_row:max_row+1, min_col:max_col+1]
    overlap = np.sum((region_mask == 255) & (region_current_mask == 1))
    total_pixels = np.sum(region_current_mask == 1)
    overlap_percentage = (overlap / total_pixels) * 100
    if overlap_percentage > 50:
        new.append(current_mask)

new = np.array(new)
furthest_distances = find_furthest_distance_in_each_layer(new)
furthest_distances_array = np.array(furthest_distances)
masks_filtered = new[furthest_distances_array <= 80]
masks_filtered = list(masks_filtered)
masksfinal = masks_filtered + filtered_masks
masksfinal = np.array(masksfinal)
np.save('filtered_masks.npy', masksfinal)
image_path = "modified_image_dilated.jpg"
image = Image.open(image_path)
image_np = np.array(image)

for mask in masks_filtered:
    image_np[mask == 1] = [255, 255, 255]

modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image_dilated.jpg"
modified_image.save("modified_image_dilated.jpg")
image_sk = io.imread("modified_image_dilated.jpg")
gray_sk = color.rgb2gray(image_sk)
threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value
cleaned = morphology.remove_small_objects(binary, min_size=50)
mask = np.where(cleaned == 0, 1, 0).astype(bool)
image_sk[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]
processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)
image22 = cv2.imread(r"modified_image_dilated.jpg")
image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)
mask_generator22 = SamAutomaticMaskGenerator(sam)
masks22 = mask_generator22.generate(image22)
masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
np.save("masks22.npy", masks22)
masks = np.load('masks22.npy', allow_pickle=True)
image_path = r"modified_image_dilated.jpg"
image = Image.open(image_path)
binary_masks = [mask_dict['segmentation'] for mask_dict in masks]
valid_mask_indices = []

for index, binary_mask in enumerate(binary_masks):
    furthest_distance = calculate_furthest_distance(binary_mask)
    area = np.sum(binary_mask)
    ratio = area / (furthest_distance ** 2)
    if ratio > 0.4:
        valid_mask_indices.append(index)

masks22_path = 'masks22.npy'
masks22 = np.load(masks22_path, allow_pickle=True)
image_path = r"modified_image_dilated.jpg"
image = Image.open(image_path)
image_np = np.array(image)

for idx in valid_mask_indices:
    mask = masks22[idx]['segmentation']
    image_np[mask] = [255, 255, 255]

modified_image = Image.fromarray(image_np)
save_path = 'new_modified_image.jpg'
modified_image.save(save_path)
image_sk = io.imread('new_modified_image.jpg')
gray_sk = color.rgb2gray(image_sk)
threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value
cleaned = morphology.remove_small_objects(binary, min_size=50)
mask = np.where(cleaned == 0, 1, 0).astype(bool)
image_sk[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]
processed_img_path = 'processed_image.jpg'
cv2.imwrite(processed_img_path, image_sk)
masks22 = np.delete(masks22, valid_mask_indices)
np.save("masks22.npy", masks22)
mask_file = 'masks22.npy'
masks = np.load(mask_file, allow_pickle=True)
connected_components = [np.max(label(mask['segmentation'])) if 'segmentation' in mask else 0 for mask in masks]
min_pixel_size = 15

for i, num_components in enumerate(connected_components):
    if num_components > 1:
        masks[i]['segmentation'] = remove_small_components(masks[i]['segmentation'], min_pixel_size)

np.save(mask_file, masks)
masks_path = "masks22.npy"
masks = np.load(masks_path, allow_pickle=True)
skeletonized_segmentations = process_skeletons(masks)
output_file = 'skeletonized_segmentations.npy'
np.save(output_file, skeletonized_segmentations)
disconnected_masks_indices, connectcomponents = analyze_masks_connectivity(masks)

def identify_valid_endpoints_masks(masks, min_distance=25):
    valid_endpoints_masks = []
    for index, mask_dict in enumerate(masks):
        mask = mask_dict
        skeleton = mask
        endpoints = detect_endpoints(skeleton)
        branch_points = detect_branch_points(skeleton)
        if not np.any(branch_points):
            valid_endpoints_count = np.count_nonzero(endpoints)
        else:
            distances_to_nearest_branch = endpoints_to_nearest_branch_distance(skeleton, endpoints, branch_points)
            valid_endpoints_count = np.sum(distances_to_nearest_branch > min_distance) + 0.5 * np.sum(distances_to_nearest_branch <= min_distance)
        if math.floor(valid_endpoints_count) > 2 * connectcomponents[index]:
            valid_endpoints_masks.append(index)
    return valid_endpoints_masks

valid_endpoints_masks_indices = identify_valid_endpoints_masks(skeletonized_segmentations)
skeletonized_segmentations_path = 'skeletonized_segmentations.npy'
skeletonized_segmentations = np.load(skeletonized_segmentations_path, allow_pickle=True)
masks_with_high_curvature_points = []
sharp_points_all_layers = []

for index, skeleton_layer in enumerate(skeletonized_segmentations):
    sharp_points = find_sharp_angle_points(skeleton_layer)
    if sharp_points:
        masks_with_high_curvature_points.append(index)
    sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})

final_set = set(masks_with_high_curvature_points) | set(valid_endpoints_masks_indices) | set(disconnected_masks_indices)
final = list(final_set)
image = io.imread('processed_image.jpg')
masks = np.load('masks22.npy', allow_pickle=True)
new_image = copy.deepcopy(image)
masks_length = len(masks)
new_list = list(range(masks_length))
final2 = list(set(new_list) - set(final))

for index in final2:
    mask = masks[index]['segmentation']
    new_image[mask == 1] = 255

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

masks_file = 'masks22.npy'
masks = np.load(masks_file, allow_pickle=True)

for index in final2:
    if connectcomponents[index] == 2:
        mask = masks[index]['segmentation']
        original_pixels = np.sum(mask)
        skeleton = skeletonize(mask)
        skeleton_pixels = np.sum(skeleton)
        ratio = original_pixels / skeleton_pixels if skeleton_pixels > 0 else 0
        labeled_mask, num_components = label(mask, return_num=True, connectivity=1)
        props = regionprops(labeled_mask)
        if num_components == 2:
            coords_1 = props[0].coords
            coords_2 = props[1].coords
            point_1, point_2 = find_closest_points(coords_1, coords_2)
            draw_filled_rectangle(mask, point_1, point_2, ratio)
            masks[index]['segmentation'] = mask

np.save(masks_file, masks)
masks_file = 'masks22.npy'
masks = np.load(masks_file, allow_pickle=True)

for index in final2:
    Sperm_tail.append(masks[index])

new_image_path = 'final111.jpg'
io.imsave(new_image_path, new_image)
image_sk = io.imread('final111.jpg')
gray_sk = color.rgb2gray(image_sk)
threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value
cleaned = morphology.remove_small_objects(binary, min_size=50)
mask = np.where(cleaned == 0, 1, 0).astype(bool)
image_sk[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]
processed_img_path = 'modified_image_dilated0.jpg'
cv2.imwrite(processed_img_path, image_sk)


#The fifth step:Iterative Mask Processing and Whitening
prev_count = -1
current_count = 0
stagnant_rounds = 0
X = 0
while True:
    image22 = cv2.imread(f"modified_image_dilated{X}.jpg")
    image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)
    mask_generator22 = SamAutomaticMaskGenerator(sam)
    masks22 = mask_generator22.generate(image22)
    masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
    np.save("masks22.npy", masks22)
    masks = np.load('masks22.npy', allow_pickle=True)
    if masks.size == 0:
        X -= 1
        break
    image_path = f"modified_image_dilated{X}.jpg"
    image = Image.open(image_path)
    binary_masks = [mask_dict['segmentation'] for mask_dict in masks]
    valid_mask_indices = []
    for index, binary_mask in enumerate(binary_masks):
        furthest_distance = calculate_furthest_distance(binary_mask)
        area = np.sum(binary_mask)
        ratio = area / (furthest_distance ** 2)
        if ratio > 0.2:
            valid_mask_indices.append(index)
    masks22_path = 'masks22.npy'
    masks22 = np.load(masks22_path, allow_pickle=True)
    image_path = f"modified_image_dilated{X}.jpg"
    image = Image.open(image_path)
    image_np = np.array(image)
    for idx in valid_mask_indices:
        mask = masks22[idx]['segmentation']
        image_np[mask] = [255, 255, 255]
    modified_image = Image.fromarray(image_np)
    save_path = 'new_modified_image.jpg'
    modified_image.save(save_path)
    image_sk = io.imread('new_modified_image.jpg')
    gray_sk = color.rgb2gray(image_sk)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=20)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_sk[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]
    processed_img_path = 'processed_image.jpg'
    cv2.imwrite(processed_img_path, image_sk)
    masks22 = np.delete(masks22, valid_mask_indices)
    np.save("masks22.npy", masks22)
    masks_path = "masks22.npy"
    masks = np.load(masks_path, allow_pickle=True)
    if masks.size == 0:
        X -= 1
        break
    mask_file = 'masks22.npy'
    masks = np.load(mask_file, allow_pickle=True)
    connected_components = [np.max(label(mask['segmentation'])) if 'segmentation' in mask else 0 for mask in masks]
    min_pixel_size = 10
    for i, num_components in enumerate(connected_components):
        if num_components > 1:
            masks[i]['segmentation'] = remove_small_components(masks[i]['segmentation'], min_pixel_size)
    np.save(mask_file, masks)
    masks_path = "masks22.npy"
    masks = np.load(masks_path, allow_pickle=True)
    skeletonized_segmentations = process_skeletons(masks)
    output_file = 'skeletonized_segmentations.npy'
    np.save(output_file, skeletonized_segmentations)
    disconnected_masks_indices = analyze_skeleton_connectivity(masks)
    valid_endpoints_masks_indices = identify_valid_endpoints_masks(skeletonized_segmentations)
    skeletonized_segmentations_path = 'skeletonized_segmentations.npy'
    skeletonized_segmentations = np.load(skeletonized_segmentations_path, allow_pickle=True)
    masks_with_high_curvature_points = []
    sharp_points_all_layers = []
    for index, skeleton_layer in enumerate(skeletonized_segmentations):
        sharp_points = find_sharp_angle_points(skeleton_layer)
        if sharp_points:
            masks_with_high_curvature_points.append(index)
        sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})
    final_set = set(masks_with_high_curvature_points) | set(valid_endpoints_masks_indices) | set(disconnected_masks_indices)
    final = list(final_set)
    image = io.imread('processed_image.jpg')
    masks = np.load('masks22.npy', allow_pickle=True)
    new_image = copy.deepcopy(image)
    masks_length = len(masks)
    new_list = list(range(masks_length))
    final2 = list(set(new_list) - set(final))
    for index in final2:
        mask = masks[index]['segmentation']
        new_image[mask == 1] = 255
    for index in final2:
        Sperm_tail.append(masks[index])
    new_image_path = 'final111.jpg'
    io.imsave(new_image_path, new_image)
    img_path = 'final111.jpg'
    image_sk = io.imread('final111.jpg')
    gray_sk = color.rgb2gray(image_sk)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=20)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_sk[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]
    processed_img_path = f"modified_image_dilated{X + 1}.jpg"
    cv2.imwrite(processed_img_path, image_sk)
    current_count = len(Sperm_tail)
    if current_count == prev_count:
        stagnant_rounds += 1
    else:
        stagnant_rounds = 0
    prev_count = current_count
    if stagnant_rounds >= 2:
        break
    X += 1
    if X >= 6:
        X -= 1
        break


def sample_every_other_pixel(coord_list):
    """Sample every other pixel in a list of coordinates."""
    return coord_list


#The sixth step:Comprehensive Image and Mask Processing Workflow
masks_path = 'masks22.npy'
image_path = f'data/original_images/{I}.jpg'
extracted_images = extract_segments_from_image(masks_path, image_path)
image_path = f"modified_image_dilated{X+1}.jpg"
image_array = io.imread(image_path)
masks_path = 'masks22.npy'
masks_data = np.load(masks_path, allow_pickle=True)
for mask_dict in masks_data:
    mask = mask_dict['segmentation']
    image_array[mask] = [255, 255, 255]
gray_sk = color.rgb2gray(image_array)
if len(np.unique(gray_sk)) > 1:
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_array[mask] = [255, 255, 255]
whitening_mask_sk = np.all(image_array > [200, 200, 200], axis=-1)
image_array[whitening_mask_sk] = [255, 255, 255]
final_image_sk = Image.fromarray(image_array.astype(np.uint8))
processed_img_path = 'new_processed_image.jpg'
final_image_sk.save(processed_img_path)
image_path = 'new_processed_image.jpg'
image_array = io.imread(image_path)
extracted_images_on_canvas = extract_and_place_on_canvas(image_array)
for idx, img in enumerate(extracted_images_on_canvas, start=len(extracted_images)):
    save_path = f'extracted_image{idx}_0.jpg'
    Image.fromarray(img).save(save_path)
image_path = 'new_processed_image.jpg'
image_array = io.imread(image_path)
masks_path = 'masks22.npy'
updated_masks = extract_components_and_update_masks(image_array, masks_path)
updated_masks_path = 'masks22.npy'
np.save(updated_masks_path, updated_masks)
data_path = 'masks22.npy'
data = np.load(data_path, allow_pickle=True)
skeletonized_segmentations = process_skeletons(data)
masks_with_high_curvature_points = []
sharp_points_all_layers = []
for index, skeleton_layer in enumerate(skeletonized_segmentations):
    sharp_points = find_sharp_angle_points(skeleton_layer)
    if sharp_points:
        masks_with_high_curvature_points.append(index)
    sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})
skeletons = extract_and_skeletonize(data)
points_for_sam = []
for index, skeleton in enumerate(skeletons):
    branch_points = find_branch_points(skeleton)
    branch_point_coordinates = set(map(tuple, np.argwhere(branch_points)))
    high_curvature_points = set()
    for layer in sharp_points_all_layers:
        if layer['layer'] == index:
            high_curvature_points = set(map(tuple, layer['sharp_points']))
            break
    combined_points = branch_point_coordinates.union(high_curvature_points)
    skeleton_without_branches = np.copy(skeleton)
    for coord in combined_points:
        skeleton_without_branches[coord] = 0
    labeled_skeleton, _ = label(skeleton_without_branches, return_num=True)
    small_components = [region for region in regionprops(labeled_skeleton) if region.area < 40]
    branch_points_list = [tuple(coord) for coord in branch_point_coordinates]
    for component in small_components:
        component_coords = component.coords
        component_mask = np.zeros_like(skeleton_without_branches, dtype=bool)
        for coord in component_coords:
            component_mask[tuple(coord)] = True
        endpoints = find_endpoints(component_mask)
        endpoints_coordinates = np.argwhere(endpoints)
        if sum(is_adjacent_to_branch(tuple(endpoint), branch_points_list) for endpoint in endpoints_coordinates) != 3:
            for coord in component_coords:
                skeleton_without_branches[tuple(coord)] = 0
    labeled_skeleton, num_components = label(skeleton_without_branches, return_num=True)
    component_dict = {}
    for i in range(1, num_components + 1):
        component_coords = np.argwhere(labeled_skeleton == i)
        component_dict[f'Component_{i}'] = sample_every_other_pixel([tuple(coord[::-1]) for coord in component_coords])
    points_for_sam.append({f'Skeleton_{index + 1}': component_dict})

endpoints_for_sam = []
for skeleton_dict in points_for_sam:
    for skeleton_key, components in skeleton_dict.items():
        skeleton_endpoints = {}
        for component_key, component_coords in components.items():
            endpoints = find_endpoints_of_components(component_coords)
            skeleton_endpoints[component_key] = endpoints
        endpoints_for_sam.append({skeleton_key: skeleton_endpoints})

sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for i, skeleton in enumerate(points_for_sam):
    for skeleton_name, components in skeleton.items():
        for component_name, coordinates in components.items():
            item = coordinates
            image = cv2.imread(f'extracted_image{i}_0.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            input_point = np.array(item)
            input_label = np.array([1] * len(input_point))
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            Sperm_tail.append(masks[0])

Sperm_tail = standardize_format(Sperm_tail)
Sperm_tail = np.array(Sperm_tail)
np.save('Sperm_tail.npy', Sperm_tail)
file_path = 'Sperm_tail.npy'
original_masks = np.load(file_path, allow_pickle=True)
filtered_masks = filter_masks(original_masks, threshold=450)
np.save('Sperm_tail.npy', np.array(filtered_masks, dtype=object))

    
#The seventh step:
masks_path = 'Sperm_tail.npy'
masks = np.load(masks_path, allow_pickle=True)
masks_to_remove = set()
for i in range(len(masks)):
    for j in range(i + 1, len(masks)):
        iou = compute_iou(masks[i]['segmentation'], masks[j]['segmentation'])
        if iou >= 0.5:
            sum_i = masks[i]['segmentation'].sum()
            sum_j = masks[j]['segmentation'].sum()
            masks_to_remove.add(j if sum_i >= sum_j else i)

masks = np.delete(masks, list(masks_to_remove))
updated_masks_path = 'Sperm_tail.npy'
np.save(updated_masks_path, masks)
masks_path = 'filtered_masks.npy'
masks = np.load(masks_path, allow_pickle=True)
masks_to_remove = set()
for i in range(masks.shape[0]):
    for j in range(i + 1, masks.shape[0]):
        iou = compute_iou(masks[i], masks[j])
        if iou >= 0.5:
            sum_i = masks[i].sum()
            sum_j = masks[j].sum()
            masks_to_remove.add(j if sum_i >= sum_j else i)

filtered_masks = np.delete(masks, list(masks_to_remove), axis=0)
updated_masks_path = 'filtered_masks.npy'
np.save(updated_masks_path, filtered_masks)
data = np.load('Sperm_tail.npy', allow_pickle=True)
masks_to_remove = process_masks(data)
data = np.delete(data, masks_to_remove, axis=0)
np.save('Sperm_tail.npy', data)
file_path = 'filtered_masks.npy'
masks = np.load(file_path)
layers_with_multiple_components = []
for layer_index in range(masks.shape[0]):
    _, num_components = label_connected_components(masks[layer_index])
    if num_components >= 2:
        layers_with_multiple_components.append(layer_index)
new_masks = []
for layer_index in range(masks.shape[0]):
    if layer_index in layers_with_multiple_components:
        new_layers = split_layer_into_components(masks[layer_index])
        new_masks.extend(new_layers)
    else:
        new_masks.append(masks[layer_index])
new_masks = np.array(new_masks)
updated_file_path = 'filtered_masks.npy'
np.save(updated_file_path, new_masks)
file_path = 'filtered_masks.npy'
masks = np.load(file_path)
filtered_masks = [mask for mask in masks if np.sum(mask) >= 700]
np.save(file_path, filtered_masks)
masks = np.load('filtered_masks.npy')
canvases = []
for i, mask in enumerate(masks):
    canvas, line = find_and_draw_longest_line(mask)
    if canvas is not None:
        canvases.append(canvas)
final_canvas = np.zeros_like(masks[0])
for canvas in canvases:
    final_canvas = cv2.bitwise_or(final_canvas, canvas)
output_path = 'final_canvas.jpg'
cv2.imwrite(output_path, final_canvas)
file_path = 'filtered_masks.npy'
masks = np.load(file_path)
filtered_masks = remove_masks_with_border_objects(masks)
output_file_path = 'filtered_masks.npy'
np.save(output_file_path, filtered_masks)
masks = np.load('filtered_masks.npy')
longest_lines_info = []
for i, mask in enumerate(masks):
    longest_line = find_longest_line(mask)
    if longest_line:
        longest_lines_info.append({'layer': i, 'endpoints': [longest_line[0], longest_line[1]]})
sperm_tail_data = np.load('Sperm_tail.npy', allow_pickle=True)
skeletonized_segmentations = process_skeletons(sperm_tail_data)
output_file = 'skeletonized_segmentations.npy'
np.save(output_file, skeletonized_segmentations)
all_endpoints = find_modified_endpoints(skeletonized_segmentations)
for layer_data in all_endpoints:
    if len(layer_data['endpoints']) > 1:
        layer_data['endpoints'] = find_farthest_points(layer_data['endpoints'])
distance = 5
all_endpoints_20 = extend_endpoints_on_skeletons(skeletonized_segmentations, all_endpoints, distance)
image_path = f'data/original_images/{I}.jpg'
image = Image.open(image_path).convert('RGB')
image_array = np.array(image)
hsv_image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
all_endpoints = keep_corresponding_minimum_value_point(all_endpoints, all_endpoints_20, hsv_image_array)
angles_with_20_points = []
for layer_info in all_endpoints:
    layer = layer_info['layer']
    angles = []
    for endpoint in layer_info['endpoints']:
        endpoint_rc = (endpoint[1], endpoint[0])
        points = find_skeleton_points(skeletonized_segmentations[layer], endpoint_rc)
        angle = calculate_angle2(points)
        angles.append(angle)
    angles_with_20_points.append({'layer': layer, 'angles': angles})
selected_tails = find_matching_tails(longest_lines_info, all_endpoints, 27)
heads = {item['layer']: item['endpoints'] for item in longest_lines_info}
tails = {item['layer']: item['endpoints'] for item in all_endpoints}
final_selection3 = {}
for head_layer, tail_layers in selected_tails.items():
    head = heads[head_layer]
    min_angle_diff = float('inf')
    selected_tail = None
    for tail_layer in tail_layers:
        tail = tails[tail_layer]
        closest_head_point, closest_tail_point = closest_points(head, tail)
        for h_point in head:
            if h_point != closest_head_point:
                angle_diff = angle_difference(closest_head_point, h_point, tail_layer, angles_with_20_points)
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    selected_tail = tail_layer
    if min_angle_diff > 40:
        final_selection3[head_layer] = None
    else:
        final_selection3[head_layer] = selected_tail
for head_layer in heads:
    if head_layer not in final_selection3:
        final_selection3[head_layer] = None
while True:
    tail_to_heads = {}
    for head, tail in final_selection3.items():
        if tail is not None:
            tail_to_heads.setdefault(tail, []).append(head)
    conflict_tails = {tail: heads for tail, heads in tail_to_heads.items() if len(heads) > 1}
    if not conflict_tails:
        break
    for tail, conflicting_heads in conflict_tails.items():
        min_angle = float('inf')
        selected_head = None
        for head in conflicting_heads:
            head_points = heads[head]
            tail_point = tails[tail][0]
            for head_point in head_points:
                angle = angle_between_points(head_point, tail_point, (head_point[0] + 1, head_point[1]))
                if angle < min_angle:
                    min_angle = angle
                    selected_head = head
        for head in conflicting_heads:
            if head != selected_head:
                alternative_tails = [t for t in selected_tails[head] if t != tail and t not in tail_to_heads]
                final_selection3[head] = alternative_tails[0] if alternative_tails else None


#The eighth step:Final Mask Combination and JSON Export Workflow
head_masks_path = 'filtered_masks.npy'
head_masks = np.load(head_masks_path)
sperm_tail_path = 'Sperm_tail.npy'
sperm_tail = np.load(sperm_tail_path, allow_pickle=True)
segmentations = [segment['segmentation'] for segment in sperm_tail]
final_selection_filtered = {k: v for k, v in final_selection3.items() if v is not None}
combined_masks = {}
original_head_masks = []
original_tail_masks = []
for i, (head_layer, tail_layer) in enumerate(final_selection_filtered.items()):
    head_mask = head_masks[head_layer]
    tail_mask = segmentations[tail_layer]

    original_head_masks.append(head_mask)
    original_tail_masks.append(tail_mask)

    combined_mask = np.maximum(head_mask, tail_mask)
    combined_masks[f'Head_{head_layer}_Tail_{tail_layer}'] = combined_mask
np.save('combined_masks.npy', combined_masks)
np.save('original_head_masks.npy', original_head_masks)
np.save('original_tail_masks.npy', original_tail_masks)
from skimage.draw import polygon
from scipy.spatial import distance
from skimage.measure import label, regionprops
file_path = 'combined_masks.npy'
masks = np.load(file_path, allow_pickle=True)
content = masks.item()
updated_content = update_masks_with_rectangles(content)
updated_file_path = 'combined_masks.npy'
np.save(updated_file_path, updated_content)
data = np.load('combined_masks.npy', allow_pickle=True)
item = data.item()
all_values = []
if isinstance(item, dict):
    all_values.extend(item.values())
elif isinstance(item, (list, tuple)):
    for sub_item in item:
        if isinstance(sub_item, dict):
            all_values.extend(sub_item.values())
new_array = np.array(all_values, dtype=object)
valid_masks_path = 'valid_masks.npy'
valid_masks = np.load(valid_masks_path, allow_pickle=True)
if valid_masks.ndim == 0:
    valid_masks = valid_masks.item()
valid_masks_converted = np.array(valid_masks).astype(int)
Last = list(valid_masks_converted) + list(new_array)
np.save("combined_masks.npy", Last)
npy_file_path = 'combined_masks.npy'
masks = np.load(npy_file_path, allow_pickle=True)
json_output = {
    "version": "0.2.4",
    "flags": {},
    "shapes": [],
    "imagePath": f"{I}.jpg",
    "imageData": None,
    "imageHeight": 540,
    "imageWidth": 720,
    "text": ""
}
for index, mask in enumerate(masks):
    polygons = mask_to_polygon_points(mask)
    for polygon in polygons:
        shape_data = {
            "label": "sperm",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        json_output['shapes'].append(shape_data)
json_output_path = f'data/original_images/{I}.json'
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)
json_file_path = f'data/original_images/{I}.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
data['shapes'] = [shape for shape in data['shapes'] if len(shape['points']) >= 10]
with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)
print("Successfully running, please go to data/original_images to find your JSON file with the same name.")



files_to_delete = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg") + glob.glob("*.npy")
for file in files_to_delete:
    os.remove(file)
