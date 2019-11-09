#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:28:34 2019

@author: clytie
"""

import time
import subprocess

from dist_rl.wrapper import AgentWrapper


class Raiden2(object):
    def __init__(self, port=6666, with_stack=True, num_player=1, num_frames=4):
        self.port = port
        self.game = "raiden2dx"
        self.with_stack = with_stack
        self.num_player = num_player
        self.num_frames = num_frames
        self.controller = "raiden2" if with_stack else "raiden2dx"
        self.num_srd = 1
        self.processes = []

        self.agent = AgentWrapper(port)
        self.agent.setDaemon(True)
        self.agent.start()
    
    def _start_env(self):
        worker_cmd = f'''export PORT={self.port}; export NUM_FRAMES={self.num_frames}; export NUM_PLAYER={self.num_player}; ./mame64 {self.game} -le_options {self.controller}_controller.py'''
        self.processes.append(subprocess.Popen(worker_cmd, shell=True))
    
    def _get_srd(self):
        env_ids, states, rewards, dones = self.agent.get_srd_batch(self.num_srd)
        return env_ids, states, rewards, dones

    def start(self):
        if not self.processes:
            self._start_env()
        return self._get_srd()
    
    def close(self):
        if self.processes:
            subprocess.Popen(f"python kill_manager.py {self.controller}_controller.py", shell=True)
            time.sleep(0.5)
            subprocess.Popen("python kill_manager.py xvfb-run", shell=True)
            time.sleep(0.5)
            self.processes = []

    def step(self, env_ids, actions):
        assert len(actions) == self.num_srd
        self.agent.put_a_batch(env_ids, actions)
        return self._get_srd()


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    
    from tqdm import tqdm
    
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_float('temperature', 1.0, 'random rate')
    flags.DEFINE_integer('frame_stack', 4, 'number of frame')
    flags.DEFINE_string('algorithm', 'basic_ppo', 'ppo algorithm')
    flags.DEFINE_string('save_path', 'basic_ppo_log', 'save path')
    
    algorithm = FLAGS.algorithm
    save_path = FLAGS.save_path
    temperature = FLAGS.temperature
    
    action_space = 4
    frame_stack = FLAGS.frame_stack
    size = 84
    state_space = (size, size, frame_stack)
    
    if algorithm == 'tsarppo':
        from algorithms.tsarppo import TSARPPO
        ppo = TSARPPO(action_space, state_space,
                      temperature=temperature,
                      save_path=save_path)
    elif algorithm == 'tsappo':
        from algorithms.tsappo import TSAPPO
        ppo = TSAPPO(action_space, state_space,
                     temperature=temperature,
                     save_path=save_path)
    elif algorithm == 'sappo':
        from algorithms.sappo import SAPPO
        ppo = SAPPO(action_space, state_space,
                    temperature=temperature,
                    save_path=save_path)
    elif algorithm == 'sappov2':
        from algorithms.sappo_v2 import SAPPOV2
        ppo = SAPPOV2(action_space, state_space,
                      temperature=temperature,
                      save_path=save_path)
    else:
        from algorithms.basic_ppo import BasicPPO
        ppo = BasicPPO(action_space, state_space,
                       temperature=temperature,
                       save_path=save_path)
    
    env = Raiden2(num_frames=frame_stack)
    env_ids, states, rewards, dones = env.start()
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(1024)):
            actions = ppo.get_action(np.asarray(states))
            actions = [(action, 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        
    env.close()


