#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import logging

from datasets.process import get_affine_transform
from datasets.transforms import build_transforms
from datasets.process import get_final_preds
from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from utils.utils_bbox import box2cs
from utils.common import INFERENCE_PHASE

# Please make sure that root dir is the root directory of the project
root_dir = os.path.abspath('./')


def set_args(args):
    args.rootDir = root_dir
    args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
    args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
    return args


cfg = None
args = None


def get_pose_model(args):
    cfg = get_cfg(args)
    update_config(cfg, args)
    checkpoint_dict = torch.load(args.weight)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    new_model = build_model(cfg, INFERENCE_PHASE)
    new_model.load_state_dict(model_state_dict)
    return new_model

image_transforms = build_transforms(None, INFERENCE_PHASE)
image_size = np.array([288, 384])
aspect_ratio = image_size[0] * 1.0 / image_size[1]


def image_preprocess(image, prev_image, next_image, center, scale):
    trans_matrix = get_affine_transform(center, scale, 0, image_size)
    image_data = image
    image_data = cv2.warpAffine(image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    image_data = image_transforms(image_data)
    if prev_image is None or next_image is None:
        return image_data
    else:
        prev_image_data = prev_image
        prev_image_data = cv2.warpAffine(prev_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        prev_image_data = image_transforms(prev_image_data)

        next_image_data = next_image
        next_image_data = cv2.warpAffine(next_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        next_image_data = image_transforms(next_image_data)

        return image_data, prev_image_data, next_image_data


def inference_PE_batch(model, device, input_image_list: list, prev_image_list: list, next_image_list: list, bbox_list: list):
    """
        input_image : input image path
        prev_image : prev image path
        next_image : next image path
        inference pose estimation
    """
    videos_size = len(input_image_list)

    batch_input = []
    batch_margin = []
    batch_center = []
    batch_scale = []
    pose_results = []
    
    for video_index in range(videos_size):
        pose_results.append([])
        input_image = input_image_list[video_index]
        prev_image = prev_image_list[video_index]
        next_image = next_image_list[video_index]
        
        for info in bbox_list[video_index]:
            bbox = info['bbox']
            
            center, scale = box2cs(bbox[0:4], aspect_ratio)
            batch_center.append(center)
            batch_scale.append(scale)
            
            target_image_data, prev_image_data, next_image_data = image_preprocess(input_image, prev_image, next_image, center, scale)
            
            target_image_data = target_image_data.unsqueeze(0)
            prev_image_data = prev_image_data.unsqueeze(0)
            next_image_data = next_image_data.unsqueeze(0)

            one_sample_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).to(device)
            margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).to(device)

            batch_input.append(one_sample_input)
            batch_margin.append(margin)
    
    if len(batch_input) == 0:
        return pose_results
    
    batch_input = torch.cat(batch_input, dim=0).to(device)
    batch_margin = torch.cat(batch_margin, dim=0).to(device)
    # concat_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).to(device)
    # margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).to(device)
    model.eval()

    predictions = model(batch_input, margin=batch_margin)

    pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), batch_center, batch_scale)
    pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)
    
    current_video = 0
    change_idx = 0
    for pose_idx, pose in enumerate(pred_keypoints):
        video_offset = pose_idx - change_idx
        if video_offset >= len(bbox_list[current_video]):
            video_offset = 0
            current_video += 1
            change_idx = pose_idx
        
        pose_result = bbox_list[current_video][video_offset].copy()
        pose_result['keypoints'] = pose
        pose_results[current_video].append(pose_result)

    return pose_results
