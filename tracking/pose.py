#
#
#

import os
import cv2
import numpy as np
import torch
import dask
from numba import njit

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
        idx_offset = []
    
        for video_index in range(videos_size):
            pose_results.append([None] * len(bboxes[video_index]))
            input_image = curr_frames[video_index]
            prev_image = prev_frames[video_index]
            next_image = next_frames[video_index]
        
            for idx, info in enumerate(bboxes[video_index]):
                #bbox = info['bbox']
                bbox = info[1]
                idx_offset.append((video_index, idx))
            
                center, scale = box2cs(bbox[0:4], self.aspect_ratio)
                batch_center.append(center)
                batch_scale.append(scale)
            
                target_image_data, prev_image_data, next_image_data = \
                    self.preprocess_image(input_image, prev_image, next_image, center, scale)
                    
                target_image_data = dask.delayed(torch.unsqueeze)(target_image_data, 0)
                prev_image_data = dask.delayed(torch.unsqueeze)(prev_image_data, 0)
                next_image_data = dask.delayed(torch.unsqueeze)(next_image_data, 0)
                
                one_sample_input = dask.delayed(torch.cat)([target_image_data, prev_image_data, next_image_data])
                margin = dask.delayed(torch.stack)([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1)
                #margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1)

                batch_input.append(one_sample_input)
                batch_margin.append(margin)
        
        batch_input = dask.compute(*batch_input)
        batch_margin = dask.compute(*batch_margin)
    
        if len(batch_input) == 0:
            return pose_results
    
        batch_input = torch.cat(batch_input, dim=0).to(self.device)
        batch_margin = torch.cat(batch_margin, dim=0).to(self.device)

        self.model.eval()

        predictions = self.model(batch_input, margin=batch_margin)

        pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), batch_center, batch_scale)
        pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)
        
        return PoseEstimator.postprocess_pose(pose_results, pred_keypoints, bboxes, idx_offset)

    @staticmethod
    #@njit(parallel=True)
    def postprocess_pose(results: list, poses: list, bboxes: list, idx_offset: list):
        for pose, (idx, offset) in zip(poses, idx_offset):
            res = dict()
            res['track_id'] = bboxes[idx][offset][0]
            res['bbox'] = bboxes[idx][offset][1]
            res['keypoints'] = pose
            results[idx][offset] = res
    
        return results