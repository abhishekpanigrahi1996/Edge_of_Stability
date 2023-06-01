
   
import os,sys
import random,math
import numpy as np
import torch
from typing import Any
from typing import Callable
from typing import Optional
import torch.utils.data.dataset as dataset
import argparse
import copy

from torch.utils.data import Sampler
import attr
import torch
import torchvision


class My_BatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, drop_last, sample_mode, num_replicas = 1 ) -> None:
        assert(drop_last or  (sample_mode in ['random_shuffling']))
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sample_mode = sample_mode
        self.num_replicas = num_replicas
        self.index_list = None
            
    def __iter__(self):
        if self.sample_mode == 'random_shuffling':
            self.index_list = torch.randperm(self.dataset_size).tolist()
            for i in range(len(self)):
                yield self.index_list[i*self.batch_size:(i+1)*self.batch_size]
## if index exceeds self.dataset_size, the batch will be truncated automatically
        elif self.sample_mode == 'without_replacement':
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size, replace=False))
        elif self.sample_mode == 'with_replacement':
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size, replace=True))
        elif self.sample_mode == 'fixed_sequence': ##must drop last
#             assert(self.dataset_size % self.batch_size == 0)
            if self.index_list is None:
                self.index_list = torch.randperm(self.dataset_size).tolist()
            for i in range(len(self)):
                yield self.index_list[i*self.batch_size:(i+1)*self.batch_size]
        elif self.sample_mode == 'two_without_replacement': ##must drop last
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size//2, replace=False))+ list(np.random.choice(self.dataset_size,self.batch_size//2, replace=False))
                
    def __len__(self):
        if self.drop_last:
            return self.dataset_size // (self.batch_size * self.num_replicas) # type: ignore
        else:
            return (self.dataset_size + self.batch_size * self.num_replicas - 1) // (self.batch_size * self.num_replicas)  # type: ignore

        
        