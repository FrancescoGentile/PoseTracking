#
#
#

import sys
sys.path.insert(1, '../')

import os
import cv2
import numpy as np
import torch
import dask

from datasets.process import get_affine_transform
from datasets.transforms import build_transforms
from datasets.process import get_final_preds
from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from utils.utils_bbox import box2cs
from utils.common import INFERENCE_PHASE


class PoseEstimator: 
    
    def __init__(self, root_dir: str, config_file: str, weights: str, device: str) -> None:
        self.model = self.create_model(root_dir, config_file, weights)
        self.model = self.model.to(device)
        
        self.device = device
        
        self.image_transforms = build_transforms(None, INFERENCE_PHASE)
        self.image_size = np.array([288, 384])
        self.aspect_ratio = self.image_size[0] * 1.0 / self.image_size[1]
    
    def create_model(self, root_dir: str, config_file: str, weights: str): 
        config_file = os.path.abspath(os.path.join(root_dir, config_file))
        weights = os.path.abspath(os.path.join(root_dir, weights))
        
        cfg = get_cfg()
        update_config(cfg, root_dir, config_file)
        
        checkpoint_dict = torch.load(weights)
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
        model = build_model(cfg, INFERENCE_PHASE)
        model.load_state_dict(model_state_dict)
        
        return model

    def preprocess_image(self, image, prev_image, next_image, center, scale):
        trans_matrix = dask.delayed(get_affine_transform)(center, scale, 0, self.image_size)
        
        image = dask.delayed(cv2.warpAffine)(image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        image = dask.delayed(self.image_transforms)(image)
        
        prev_image = dask.delayed(cv2.warpAffine)(prev_image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        prev_image = dask.delayed(self.image_transforms)(prev_image)

        next_image = dask.delayed(cv2.warpAffine)(next_image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        next_image = dask.delayed(self.image_transforms)(next_image)

        return image, prev_image, next_image
    
    def estimate(self, prev_frames: list, curr_frames: list, next_frames: list, bboxes: list): 
        videos_size = len(curr_frames)

        batch_input = []
        batch_margin = []
        batch_center = []
        batch_scale = []
        pose_results = []
    
        for video_index in range(videos_size):
            pose_results.append([])
            input_image = curr_frames[video_index]
            prev_image = prev_frames[video_index]
            next_image = next_frames[video_index]
        
            for info in bboxes[video_index]:
                bbox = info['bbox']
            
                center, scale = box2cs(bbox[0:4], self.aspect_ratio)
                batch_center.append(center)
                batch_scale.append(scale)
            
                target_image_data, prev_image_data, next_image_data = \
                    self.preprocess_image(input_image, prev_image, next_image, center, scale)
                    
                target_image_data = dask.delayed(torch.unsqueeze)(target_image_data, 0)
                prev_image_data = dask.delayed(torch.unsqueeze)(prev_image_data, 0)
                next_image_data = dask.delayed(torch.unsqueeze)(next_image_data, 0)
                
                one_sample_input = dask.delayed(torch.cat)([target_image_data, prev_image_data, next_image_data])
                margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1)

                batch_input.append(one_sample_input)
                batch_margin.append(margin)
        
        batch_input = dask.compute(*batch_input)
    
        if len(batch_input) == 0:
            return pose_results
    
        batch_input = torch.cat(batch_input, dim=0).to(self.device)
        batch_margin = torch.cat(batch_margin, dim=0).to(self.device)

        self.model.eval()

        predictions = self.model(batch_input, margin=batch_margin)

        pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), batch_center, batch_scale)
        pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)
    
        current_video = 0
        change_idx = 0
        pose_idx = 0
        while pose_idx < len(pred_keypoints):
            pose = pred_keypoints[pose_idx]

            video_offset = pose_idx - change_idx
            if video_offset >= len(bboxes[current_video]):
                video_offset = 0
                current_video += 1
                change_idx = pose_idx
                continue 
            else:
                pose_result = bboxes[current_video][video_offset].copy()
                pose_result['keypoints'] = pose
                pose_results[current_video].append(pose_result)
                pose_idx += 1

        return pose_results