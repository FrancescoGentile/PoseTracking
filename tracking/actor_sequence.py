#
#
#

from tqdm import tqdm
import mmcv
import os
import json
import numpy as np
import dask
from dask.distributed import Client, Variable, Lock, worker_client

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class VideoActor:
    def __init__(self, input: str, output: str, labels: str) -> None:
        '''
        Parameters:
        - input: directory containing the files
        - output: directory where to save the file
        - labels: file containing the label for each file
        '''
        self.input = input
        self.files = os.listdir(input)
        self.output = output
        
        with open(labels, 'rb') as f:
            self.labels = json.load(f)
        
        self.init_video()
        
    def init_video(self):
        with Lock('counter-lock'):
            counter = Variable('counter')
            self.idx = counter.get()
            counter.set(self.idx + 1)
        
        if self.idx >= len(self.files):
            self.video = None
            return
        
        self.current = 0
        self.filename = self.files[self.idx]
        self.video = mmcv.VideoReader(os.path.join(self.input, self.filename), 3)
        video_id = self.filename[0:11]
        self.data = { 'video_id': video_id, 
                      'label': self.labels[video_id]['annotations']['label'], 
                      'frames': [] }
        self.tqdm = tqdm(total=len(self.video), desc=self.filename, position=self.idx, ncols=100, initial=0)
    
    def get_frames(self): 
        curr = self.video[self.current]
        prev = self.video[self.current - 1] if self.current > 0 else curr
        nxt = self.video[self.current + 1] if self.current < len(self.video) - 1 else curr
        
        self.current += 1
        self.tqdm.update(1)
        
        return prev, curr, nxt
    
    def next(self):
        if self.current >= len(self.video):
            self.save_video()
            self.init_video()
            if self.video is None:
                return
        
        return self.get_frames(), self.idx
            
    
    def add_pose(self, pose): 
        self.data['frames'].append({ 'frame_id': self.idx - 1, 'people': pose })
    
    def save_video(self):
        output_file = os.path.join(self.output, self.filename)
        with open(output_file, 'w') as f:
            json.dump(self.data, f, cls=NumpyArrayEncoder)

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
        self.labels = labels

        self.num_actors = min(batch, len(os.listdir(input)))
        #self.client = Client(set_as_default=True, 
        #                     processes=True, 
        #                     direct_to_workers=True
        #                     #n_workers=self.num_actors
        #                    )
    
    def __iter__(self):
        counter = Variable('counter')
        counter.set(0)
        
        #self.actors = [
        #    self.client.submit(
        #        VideoActor, self.input, self.output, self.labels, actor=True).result() 
        #    for _ in range(self.num_actors)]
        
        self.actors = [
            VideoActor(self.input, self.output, self.labels)
            for _ in range(self.num_actors)]

        return self
    
    def __next__(self):
        '''
        futures = []
        with worker_client() as client:
            for worker in self.actors: 
                #futures.append(dask.delayed(VideoActor.next)(worker))
                futures.append(client.submit(VideoActor, worker))
            
            results = client.gather(futures)
        
        #results = [future.result() for future in futures]
        #results = dask.compute(*futures)
        '''
        
        results = [actor.next() for actor in self.actors]
        
        if len(results) == 0:
            raise StopIteration
        
        i = 0
        while i < len(results):
            if results[i] is None:
                del results[i]
                del self.actors[i]
                continue
            i += 1
        
        return results
    
    def add_poses(self, poses): 
        futures = []
        for actor, pose in zip(self.actors, poses):
            #futures.append(actor.add_pose(pose))
            actor.add_pose(pose)
        
        '''
        for future in futures:
            future.result()
        '''