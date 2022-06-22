#
#
#

import sys
sys.path.insert(1, '../')

import torch
import numpy as np
import dask

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.tracker.byte_tracker import BYTETracker

MIN_BOX_AREA = 10
ASPECT_RATIO_THRESHOLD = 1.6

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        decoder=None,
        device=torch.device('cuda:0'),
        fp16=True
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, images):
        infos = []
        images_proc = []
        for img in images:
            height, width = img.shape[:2]
            infos.append([height, width])
            #img = dask.delayed(preproc)(img, self.test_size, self.rgb_means, self.std)
            img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
            images_proc.append(img)
        
        #images_proc = dask.compute(*images_proc)
        #images_proc = [img for img, _ in images_proc]
        images_proc = np.array(images_proc)
        torch_images = torch.from_numpy(images_proc).float().to(self.device)
        
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

class Detector: 
    
    def __init__(self, exp_file: str, checkpoint: str, device: str) -> None:
        self.predictor = self.create_predictor(exp_file, checkpoint, device)
        self.trackers = {}
        
    def create_predictor(self, exp_file: str, checkpoint: str, device: str): 
        self.exp = get_exp(exp_file, None)

        model = self.exp.get_model().to(device)
        model.eval()

        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model = fuse_model(model)
        model = model.half()  # to FP16

        return Predictor(model, self.exp, None, device, True)
    
    def get_trackers(self, ids: list) -> 'list[BYTETracker]':
        res = []
        trackers = {}
        for id in ids:
            if id in self.trackers: 
                trackers[id] = self.trackers[id]
                res.append(self.trackers[id])
            else:
                trackers[id] = BYTETracker()
                res.append(trackers[id])
        
        self.trackers = trackers
        
        return res
    
    def detect(self, frames: list, ids: list):
        trackers = self.get_trackers(ids)
        results = []
        outputs, infos = self.predictor.inference(frames)
        
        for output, info, tracker in zip(outputs, infos, trackers):
            res = get_bboxes(output, info, tracker, self.exp.test_size)
            results.append(res)
        
        return results

def get_bboxes(output, info, tracker, test_size):
    res = []
    if output != None:
        online_targets = tracker.update(output, [info[0], info[1]], test_size)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > ASPECT_RATIO_THRESHOLD
            if tlwh[2] * tlwh[3] > MIN_BOX_AREA and not vertical:
                person = {}
                person['track_id'] = int(tid)
                person['bbox'] = np.append(tlwh, t.score)
                res.append(person)
    
    return res
    