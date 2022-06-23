#
#
#

import os
import cv2
import numpy as np
import torch

from datasets.process import get_affine_transform
from datasets.transforms import build_transforms
from datasets.process import get_final_preds
from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from utils.utils_bbox import box2cs
from utils.common import INFERENCE_PHASE

class PoseEstimator: 
    
    def __init__(self, root_dir: str, config_file: str, weights: str, device: str, maxp: int) -> None:
        self.model = self.create_model(root_dir, config_file, weights)
        self.model = self.model.to(device)
        
        self.device = device
        self.maxp = maxp
        
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
        model.eval()
        
        return model

    def preprocess_image(self, image, prev_image, next_image, center, scale):
        trans_matrix = get_affine_transform(center, scale, 0, self.image_size)
        
        image = cv2.warpAffine(image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        image = self.image_transforms(image)
        
        prev_image = cv2.warpAffine(prev_image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        prev_image = self.image_transforms(prev_image)

        next_image = cv2.warpAffine(next_image, trans_matrix, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        next_image = self.image_transforms(next_image)
        ''''''

        return image, prev_image, next_image
    
    def preprocess(self, prev_frames: list, curr_frames: list, next_frames: list, bboxes: list):
        videos_size = len(curr_frames)
        
        total = 0
        for b in bboxes: 
            total += len(b)

        batch_input = [None] * total
        batch_margin = [None] * total
        batch_center = [None] * total
        batch_scale = [None] * total
        idx_offset = [None] * total
        pose_results = [None] * videos_size
        
        total = 0
        for video_index in range(videos_size):
            pose_results[video_index] = [None] * len(bboxes[video_index])
            input_image = curr_frames[video_index]
            prev_image = prev_frames[video_index]
            next_image = next_frames[video_index]
        
            for idx, info in enumerate(bboxes[video_index]):
                bbox = info[1]
                idx_offset[total] = (video_index, idx)
            
                center, scale = box2cs(bbox[0:4], self.aspect_ratio)
                batch_center[total] = center
                batch_scale[total] = scale
            
                target_image_data, prev_image_data, next_image_data = \
                    self.preprocess_image(input_image, prev_image, next_image, center, scale)
                
                target_image_data = torch.unsqueeze(target_image_data, 0)
                prev_image_data = torch.unsqueeze(prev_image_data, 0)
                next_image_data = torch.unsqueeze(next_image_data, 0)
                
                one_sample_input = torch.cat([target_image_data, prev_image_data, next_image_data])
                margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1)

                batch_input[total] = one_sample_input
                batch_margin[total] = margin
                
                total += 1
        
        return batch_input, batch_margin, batch_center, batch_scale, pose_results, idx_offset
    
    def estimate(self, prev_frames: list, curr_frames: list, next_frames: list, bboxes: list): 

        batch_input, batch_margin, batch_center, batch_scale, pose_results, idx_offset = \
            self.preprocess(prev_frames, curr_frames, next_frames, bboxes)
    
        num = len(batch_input)
        for i in range(0, num, self.maxp):
            input = torch.cat(batch_input[i:min(i+self.maxp, num)], dim=0).to(self.device)
            margin = torch.cat(batch_margin[i:min(i+self.maxp, num)], dim=0).to(self.device)

            with torch.no_grad():
                predictions = self.model(input, margin=margin)
            
                pred_joint, pred_conf = get_final_preds(
                    predictions.cpu().numpy(), batch_center[i:min(i+self.maxp, num)], batch_scale[i:min(i+self.maxp, num)])
                pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)
        
            PoseEstimator.postprocess_pose(pose_results, pred_keypoints, bboxes, idx_offset[i:min(i+self.maxp, num)])
        
        return pose_results

    @staticmethod
    def postprocess_pose(results: list, poses: list, bboxes: list, idx_offset: list):
        for pose, (idx, offset) in zip(poses, idx_offset):
            res = dict()
            res['track_id'] = bboxes[idx][offset][0]
            res['bbox'] = bboxes[idx][offset][1]
            res['keypoints'] = pose
            results[idx][offset] = res