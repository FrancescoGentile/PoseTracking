#
#
#

from tqdm import tqdm
import mmcv
import os
import json
import numpy as np
import dask

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SequenceLoader: 
    
    def __init__(self, input: str, output: str, labels: str, batch: int) -> None:
        '''
        Parameters:
        - input: directory containing all the file to process
        - output: directory where to save the files
        - labels: file containing the label for each file
        - batch: number of frames to return
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
    
    def __iter__(self):
        self.next_video_idx = min(self.batch, len(self.files))
        
        self.videos = []
        for idx in range(self.next_video_idx):
            self.videos.append(self.init_video(idx))
        
        return self
    
    def __next__(self):
        index = 0
        sequences = []
        ids = []
        while index < len(self.videos):
            video = self.videos[index]
            
            if video['current'] >= len(video['reader']):
                self.save_video(video)
                
                if self.next_video_idx < len(self.files):
                    self.videos[index] = self.init_video(self.next_video_idx)
                    video = self.videos[index]
                    self.next_video_idx += 1
                else: 
                    del self.videos[index]
                    continue
            
            video['tqdm'].update(1)
            current = video['current']
            video['current'] = current + 1
            
            #sequences.append(dask.delayed(get_frames)(video['reader'], current))
            sequences.append(get_frames(video['reader'], current))
            ids.append(video['id'])
            
            index += 1
        
        #sequences = dask.compute(*sequences)
        return sequences, ids
            
    def init_video(self, idx: int):
        filename = self.files[idx]
        reader = mmcv.VideoReader(os.path.join(self.input, filename))
        video_id = filename[0:11]
        data = {'video_id': video_id, 
                'label': self.labels[video_id]['annotations']['label'], 
                'frames': [] }
            
        video = {
            'id': idx,
            'filename': filename,
            'reader': reader,
            'current': 0, 
            'data': data,
            'tqdm': tqdm(total=len(reader), desc=filename, position=idx, ncols=100, initial=0)
        }
        
        return video

    def add_poses(self, poses): 
        for video, pose in zip(self.videos, poses):
            video['data']['frames'].append({ 'frame_id': video['current'] - 1, 'people': pose })
            
    def save_video(self, video: dict):
        output_file = os.path.join(
            self.output, 
            os.path.splitext(video['filename'])[0] + '.json')
        
        with open(output_file, 'w') as f:
            json.dump(video['data'], f, cls=NumpyArrayEncoder)
            
def get_frames(reader: mmcv.VideoReader, idx: int):
    curr = reader[idx]
    prev = reader[idx - 1] if idx > 0 else curr
    nxt = reader[idx + 1] if idx < len(reader) - 1 else curr
        
    return prev, curr, nxt