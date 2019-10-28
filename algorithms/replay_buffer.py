#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:41:12 2019

@author: clytie
"""

import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)
        self.len = 0
        self._max_size = max_size
    
    def add(self, s_batch, a_batch, r_batch, d_batch):
        for states, actions, rewards, dones in zip(s_batch, a_batch, r_batch, d_batch):
            len_traj = len(dones)
            self.len += len_traj
            for i in range(len_traj):
                self.buffer.append((states[i], actions[i], rewards[i], dones[i], states[i + 1]))
        self.len = min(self.len, self._max_size)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(range(buffer_size),
                                 size=batch_size,
                                 replace=False)
        batch = [self.buffer[i] for i in index]
        states_mb = np.asarray([each[0] for each in batch])
        actions_mb = np.asarray([each[1] for each in batch])
        rewards_mb = np.asarray([each[2] for each in batch]) 
        dones_mb = np.asarray([each[3] for each in batch])
        next_states_mb = np.asarray([each[4] for each in batch])

        return states_mb, actions_mb, rewards_mb, dones_mb, next_states_mb

    def __len__(self):
        return self.len