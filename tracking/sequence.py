#
#
#

from tqdm import tqdm
import os
import json
import numpy as np
import cv2

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SequenceLoader: 
    
    def __init__(self, input: str, output: str, labels: str, batch: int, progress: bool = False) -> None:
        '''
        Parameters:
        - input: directory containing all the file to process
        - output: directory where to save the files
        - labels: file containing the label for each file
        - batch: number of frames to return
        - progress: whether to display a tqdm bar
        '''
        if not os.path.isdir(input):
            raise ValueError(f'{input} is not a directory')
        
        if not os.path.isfile(labels):
            raise ValueError(f'{input} is not a file')    
        
        if not os.path.exists(output):
            os.mkdir(output)
        
        self.input = input
        self.output = output
        self.files = os.listdir(self.input)
        
        with open(labels, 'rb') as f:
            self.labels = json.load(f)
        
        self.batch = batch
        self.progress = progress
    
    def __iter__(self):
        self.next_video_idx = min(self.batch, len(self.files))
        
        self.ids = [None] * self.next_video_idx
        self.filenames = [None] * self.next_video_idx
        self.readers = [None] * self.next_video_idx
        self.currents = [None] * self.next_video_idx
        self.terminated = [None] * self.next_video_idx
        self.datas = [None] * self.next_video_idx
        
        self.prev = [None] * self.next_video_idx
        self.curr = [None] * self.next_video_idx
        self.next = [None] * self.next_video_idx
        
        if self.progress: 
            self.tqdm = [None] * self.next_video_idx
        
        for idx in range(self.next_video_idx):
            self.init_video(idx, idx)

        return self
    
    def __next__(self):
        index = 0
        while index < len(self.ids):
            if self.terminated[index]:
                self.save_video(index)
                
                if self.next_video_idx < len(self.files):
                    self.init_video(self.next_video_idx, index)
                    self.next_video_idx += 1
                else: 
                    self.remove_video(index)
                    continue
                
            self.update_frames(index)
            index += 1
        
        return self.prev, self.curr, self.next, self.ids
    
    def init_video(self, id: int, index: int):
        filename = self.files[id]
        video_id = filename[0:11]
        reader = cv2.VideoCapture(os.path.join(self.input, filename))
        
        self.ids[index] = id
        self.filenames[index] = filename
        self.readers[index] = reader
        self.currents[index] = 0
        self.terminated[index] = False
        self.datas[index] = {
            'video_id': video_id, 
            'label': self.labels[video_id]['annotations']['label'], 
            'frames': [] 
        }
        
        _, self.prev[index] = reader.read()
        self.curr[index] = self.prev[index]
        _, self.next[index] = reader.read()
        
        if self.progress:
            self.tqdm[index] = tqdm(
                total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), desc=filename, position=id, ncols=100, initial=0)
    
    def remove_video(self, index: int):
        del self.ids[index]
        del self.filenames[index]
        del self.readers[index]
        del self.currents[index]
        del self.terminated[index]
        del self.datas[index]
        
        del self.prev[index]
        del self.curr[index]
        del self.next[index]
        
        if self.progress: 
           del self.tqdm[index]
    
    def save_video(self, index: int):
        output_file = os.path.join(
            self.output, 
            os.path.splitext(self.filenames[index])[0] + '.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.datas[index], f, cls=NumpyArrayEncoder)

    def add_poses(self, poses): 
        for data, current, pose in zip(self.datas, self.currents, poses):
            data['frames'].append({ 'frame_id': current - 1, 'people': pose })
          
    def update_frames(self, index: int):
        self.prev[index] = self.curr[index]
        self.curr[index] = self.next[index]
        
        ret, frame = self.readers[index].read()
        if ret: 
            self.next[index] = frame 
        else: 
            self.terminated[index] = True
        
        self.currents[index] += 1
        
        if self.progress:
            self.tqdm[index].update(1)

        return index
    
    
    
class MockSequenceLoader: 
    
    def __init__(self, input: str, output: str, labels: str, batch: int, progress: bool = False) -> None:
        '''
        Parameters:
        - input: directory containing all the file to process
        - output: directory where to save the files
        - labels: file containing the label for each file
        - batch: number of frames to return
        - progress: whether to display a tqdm bar
        '''
        if not os.path.isdir(input):
            raise ValueError(f'{input} is not a directory')
        
        if not os.path.isfile(labels):
            raise ValueError(f'{input} is not a file')    
        
        if not os.path.exists(output):
            os.mkdir(output)
        
        self.input = input
        self.output = output
        self.files = os.listdir(self.input)
        
        with open(labels, 'rb') as f:
            self.labels = json.load(f)
        
        self.batch = batch
        self.progress = progress
    
    def __iter__(self):
        self.next_video_idx = min(self.batch, len(self.files))
        
        self.ids = [None] * self.next_video_idx
        
        self.prev = [None] * self.next_video_idx
        self.curr = [None] * self.next_video_idx
        self.next = [None] * self.next_video_idx
        
        if self.progress: 
            self.tqdm = [None] * self.next_video_idx
        
        for idx in range(self.next_video_idx):
            self.init_video(idx, idx)

        return self
    
    def __next__(self):
        return self.prev, self.curr, self.next, self.ids
    
    def init_video(self, id: int, index: int):
        filename = self.files[id]
        reader = cv2.VideoCapture(os.path.join(self.input, filename))
    
        self.ids[index] = id
        _, self.prev[index] = reader.read()
        self.curr[index] = self.prev[index]
        _, self.next[index] = reader.read()
        
        if self.progress:
            self.tqdm[index] = tqdm(total=reader.get(
                cv2.CAP_PROP_FRAME_COUNT), desc=filename, position=index, ncols=100, initial=0)

    def add_poses(self, _): 
        pass