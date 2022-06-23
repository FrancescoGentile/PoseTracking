#
#
#

import argparse
import math
import dask
from concurrent.futures import ThreadPoolExecutor

from sequence import SequenceLoader
from detection import Detector
from pose import PoseEstimator

import nvidia_smi

def make_parser():
    parser = argparse.ArgumentParser("Pose tracking")
    
    parser.add_argument('--input', type=str, required=True, help='path to the directory with the videos')
    parser.add_argument('--labels', type=str, required=True, help='path to the json file with the labels')
    parser.add_argument('--output', default='', required=True, help='directory where to save results')
    parser.add_argument('--device', default='0', help='gpu device used for inference')
    parser.add_argument('--batch', type=int, required=True, help='number of videos processed simultaneously')

    # BYTETrack
    parser.add_argument('-f', '--exp-file', default=None, type=str, help='your experiment description file',)
    parser.add_argument('-c', '--ckpt', default=None, type=str, help='checkpoint file for BYTETrack')
    
    # DCPose
    parser.add_argument('--root-dir', type=str, default= '../', help='root directory of the project')
    parser.add_argument('--cfg', help='experiment configure file name for pose model', required=False, type=str,
                        default="./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml")
    parser.add_argument('--weights', help='pose model weight file', required=False, type=str, 
                        default='./DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth')
    
    return parser

def get_available_memory(device):
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total = info.free

    nvidia_smi.nvmlShutdown()
    
    return total

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    dask.config.set(pool=ThreadPoolExecutor())
    
    memory = get_available_memory(int(args.device))
    number = math.floor(memory / 1e9)
    args.device = 'cuda:' + args.device
    
    if args.batch > number: 
        print(f'Ideal batch size should be less than or equal to {number}')
    
    sequence_loader = SequenceLoader(args.input, args.output, args.labels, args.batch, True)
    detector = Detector(args.exp_file, args.ckpt, args.device, number)
    pose_estimator = PoseEstimator(args.root_dir, args.cfg, args.weights, args.device, number)
    
    for prev, curr, nxt, ids in sequence_loader: 
        bboxes = detector.detect(curr, ids)
        poses = pose_estimator.estimate(prev, curr, nxt, bboxes) 
        sequence_loader.add_poses(poses)
    
    print('Terminated')
        
        
if __name__ == '__main__':
    main()
