#
#
#

import argparse
import dask
from dask.distributed import Client

from sequence import SequenceLoader
from detection import Detector
from pose import PoseEstimator

def make_parser():
    parser = argparse.ArgumentParser("Pose tracking")
    
    parser.add_argument('--input', type=str, help='path to the directory with the videos')
    parser.add_argument('--labels', type=str, help='path to the json file with the labels')
    parser.add_argument('--output', default='', help='directory where to save results')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--batch', type=int, default='5', help='number of videos processed simultaneously')

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

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    dask.config.set(scheduler='threads')
    #client = Client(processes=False)
    
    sequence_loader = SequenceLoader(args.input, args.output, args.labels, args.batch)
    detector = Detector(args.exp_file, args.ckpt, args.device)
    pose_estimator = PoseEstimator(args.root_dir, args.cfg, args.weights, args.device)
    
    for sequences, ids in sequence_loader:        
        prev_frames = []
        curr_frames = []
        next_frames = []
        
        for p, c, n in sequences:
            prev_frames.append(p)
            curr_frames.append(c)
            next_frames.append(n)
        
        bboxes = detector.detect(curr_frames, ids)
        poses = pose_estimator.estimate(prev_frames, curr_frames, next_frames, bboxes)
        
        sequence_loader.add_poses(poses)
    
    print('Terminated')
        
        
if __name__ == '__main__':
    main()
