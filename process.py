# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest, tee, chain, islice
import json
from json import JSONEncoder
import os
import argparse
import pprint
import mmcv as mmcv

import torch
import numpy as np

from tqdm import tqdm

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.tracker.byte_tracker import BYTETracker

from inference import inference_PE_batch, get_pose_model, set_args

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, images):
        infos = []
        images_proc = []
        for img in images:
            height, width = img.shape[:2]
            infos.append([height, width])
            img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
            images_proc.append(img)
        
        torch_images = np.array(images_proc)
        torch_images = torch.from_numpy(torch_images).float().to(self.device)
        if self.fp16:
            torch_images = torch_images.half()

        with torch.no_grad():
            outputs = self.model(torch_images)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, infos


def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    return person_results


def make_parser():
    parser = argparse.ArgumentParser("Pose tracking")
    
    # DCPose
    parser.add_argument('--cfg', help='experiment configure file name for pose model', required=False, type=str,
                        default="./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml")
    parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
                        default='DcPose')
    parser.add_argument('-weight', help='pose model weight file', required=False, type=str
                        , default='./DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    parser.add_argument('--input', type=str, help='path to the directory with the videos')
    parser.add_argument('--labels', type=str, help='path to the json file with the labels')
    parser.add_argument('--output', default='', help='directory where to save results')
    
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    # exp file
    parser.add_argument("-f", "--exp-file", default=None, type=str, help="your experiment description file",)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.",)
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    return parser

def create_predictor(args: argparse.Namespace, exp) -> Predictor: 

    output_dir = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.trt:
        args.device = "cuda:0"
    args.device = torch.device(args.device)


    if args.conf is not None:
        exp.test_conf = args.conf

    model = exp.get_model().to(args.device)
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(output_dir, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    return predictor

def get_people_bbox(predictor: Predictor, trackers, frames, args, exp):
    results = []
    outputs, infos = predictor.inference(frames)
    for output, info, tracker in zip(outputs, infos, trackers):
        res = []
        if output == None:
            results.append(res)
            continue

        online_targets = tracker.update(output, [info[0], info[1]], exp.test_size)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                person = {}
                person['track_id'] = int(tid)
                person['bbox'] = np.append(tlwh, t.score)
                res.append(person)
        
        results.append(res)
    
    return results

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([some_iterable[0]], prevs)
    nexts = chain(islice(nexts, 1, None), [some_iterable[-1]])
    return zip(prevs, items, nexts)

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    exp = get_exp(args.exp_file, None)
    predictor = create_predictor(args, exp)
    
    args = set_args(args)
    device = torch.device(args.device)
    pose_model = get_pose_model(args).to(device)

    input_files = os.listdir(args.input)
    print(f'Found {len(input_files)} videos to process')

    with open(args.labels, 'rb') as f:
        labels = json.load(f)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)
        
    files = []
    videos = []
    frames_idx = []
    trackers = []
    outputs = []
    tqdms = []
    
    prev = []
    curr = []
    nxt = []
    
    BATCH_SIZE = 5
    
    for index in range(0, min(len(input_files), BATCH_SIZE)):
        file = input_files[index] 
        files.append(file)
        # create video
        video = mmcv.VideoReader(os.path.join(args.input, file))
        videos.append(video)
        frames_idx.append(0)
        # create tracker
        tracker = BYTETracker(args, frame_rate=30)
        trackers.append(tracker)
        # create output
        video_id = file[0:11]
        outputs.append({ 'video_id': video_id, 
                        'label': labels[video_id]['annotations']['label'], 
                        'frames': [] })

        tqdms.append(tqdm(total=len(video), desc=file, position=index, ncols=100, initial=0))
        
        prev.append(None)
        curr.append(None)
        nxt.append(None)
        
    next_video_idx = max(len(input_files), BATCH_SIZE) + 1
    
    while True:
        index = 0
        while index < len(files):
            # if video at the given index is terminated
            if frames_idx[index] >= len(videos[index]):
                file_name = os.path.splitext(files[index])[0] + '.json'
                output_file = os.path.join(args.output, file_name)
                with open(output_file, 'w') as f:
                    json.dump(outputs[index], f, cls=NumpyArrayEncoder)
                
                # check if there is another video to process
                if next_video_idx < len(input_files):
                    file = input_files[next_video_idx]
                    files[index] = file
                    # create video
                    video = mmcv.VideoReader(os.path.join(args.input, file))
                    videos[index] = video
                    frames_idx[index] = 0
                    # create tracker
                    tracker = BYTETracker(args, frame_rate=30)
                    trackers[index] = tracker
                    # create output
                    video_id = file[0:11]
                    outputs[index] = { 'video_id': video_id, 
                                      'label': labels[video_id]['annotations']['label'], 
                                      'frames': [] }
                    
                    tqdms.append(tqdm(total=len(video), desc=file, position=next_video_idx, ncols=100, initial=0))
                    
                    # update counter
                    next_video_idx += 1
                else:
                    del files[index]
                    del videos[index]
                    del frames_idx[index]
                    del trackers[index]
                    del outputs[index]
                    del prev[index]
                    del curr[index]
                    del nxt[index]
                    del tqdms[index]
                    continue 
            
            tqdms[index].update(1)
            
            video = videos[index]
            current_frame_idx = frames_idx[index]
            prev[index] = video[current_frame_idx - 1] if current_frame_idx > 0 else video[0]
            curr[index] = video[current_frame_idx]
            nxt[index] = video[current_frame_idx + 1] if current_frame_idx < len(video) - 1  else video[current_frame_idx]
            frames_idx[index] += 1
            
            index += 1
        
        if len(files) == 0: # no more files to process
            break
        
        bbox_results = get_people_bbox(predictor, trackers, curr, args, exp)
            
        pose_results = inference_PE_batch(pose_model, device, prev, curr, nxt, bbox_results)
            
        for frame_id, pose_result, output in zip(frames_idx, pose_results, outputs):
            output['frames'].append({ 'frame_id': frame_id, 'people': pose_result })
        
    '''
    for index in range(0, len(input_files), 5):
        files = input_files[index:index+5]
        videos = []
        trackers = []
        outputs = []
        
        for file in files:
            # create video
            file = os.path.join(args.input, file)
            video = mmcv.VideoReader(file)
            videos.append(video)
            # create tracker
            tracker = BYTETracker(args, frame_rate=30)
            trackers.append(tracker)
            
            outputs.append({ 'name': file[0:11], 'data': [] })
        
        prev = [video[0] for video in videos]
        curr = [video[0] for video in videos]
        for frame_id, nxt in enumerate(zip_longest(*videos, fillvalue=np.zeros((1, 1, 3), dtype=np.uint8))):
            print(f'Frame {frame_id}')
            
            bbox_results = get_people_bbox(predictor, trackers, curr, args, exp)
            
            pose_results = inference_PE_batch(pose_model, device, prev, curr, nxt, bbox_results)
            
            for pose_result, output, video in zip(pose_results, outputs, videos):
                if frame_id >= len(video):
                    continue
                
                output['data'].append(pose_result)
            
            prev = curr
            curr = nxt
                
        
        for output, file in zip(outputs, files):
            file_name = os.path.splitext(file)[0] + '.json'
            output_file = os.path.join(args.output, file_name)
            with open(output_file, 'w') as f:
                json.dump(output, f, cls=NumpyArrayEncoder)
        
        videos.clear()
        trackers.clear()
    '''

if __name__ == '__main__':
    main()
