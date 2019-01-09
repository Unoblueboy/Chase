# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:18:05 2019

@author: Natha
"""
from random import sample

import numpy as np

class ExperienceBuffer(object):
    def __init__(self, buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[0]
        self.buffer.append(experience)
        
    def sample(self, size):
        return np.array(sample(self.buffer, min(size, len(self.buffer))))