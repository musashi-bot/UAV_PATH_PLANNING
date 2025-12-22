import os
import numpy as np


# boundary - start and end points of trajectories are checked for being inside
yl, yh = 116.231761, 116.550364
xl, xh = 39.807906, 40.014366

# given duration of trajectories is s seconds, how many samples can be expect at least
sample_resolution = 10

# min to max duration of trajectories in seconds
delta_low, delta_high = 3*60*60, 10*60*60

sample_interval = 5 # in seconds (also the quanta)


class NTrajectorySet:


    def __init__(self, basedir, ) -> None:
        self.basedir = basedir
        self._make()

    def _make(self):
        #files = [os.path.join(self.basedir, i) for i in os.listdir(self.basedir) if i.endswith('.csv')]
        self.data = {i[:-4]:None for i in os.listdir(self.basedir) if i.endswith('.npy')}
        self.keys = list(self.data.keys())
        self.count = len(self.keys)

    def __call__(self, key_or_index):
        if not isinstance(key_or_index, str): key_or_index = self.keys [int(key_or_index)]
        res = self.data[key_or_index]
        if res is None: 
            self.data[key_or_index] =  self._make_npy(f'{key_or_index}.npy')
            res = self.data[key_or_index]
        return res, key_or_index
                                                                                 
    def __len__(self): return self.count

    def _make_npy(self, npy): 
        return np.load(os.path.join(self.basedir, npy))

