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
    from algorithms.ppo import PPO
    
    explore_steps = 1024
    total_updates = 1000
    save_model_freq = 30
    action_space = 4
    frame_stack = 4
    size = 84
    state_space = (size, size, frame_stack)
    
    def obs_fn():
        obs = tf.placeholder(shape=[None, *state_space], dtype=tf.float32, name="image_observation")
        return obs
    
    def model_fn(obs):
        x = tf.layers.conv2d(obs, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
    
        logit_action_probability = tf.layers.dense(
                x, action_space,
                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        state_value = tf.squeeze(tf.layers.dense(
                x, 1, kernel_initializer=tf.truncated_normal_initializer()))
        return logit_action_probability, state_value
    
    ppo = PPO(action_space, obs_fn, model_fn, temperature=0.1, train_epoch=5, batch_size=64, save_path='./raiden2_model')
    
    env = Raiden2()
    env_ids, states, rewards, dones = env.start()
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = ppo.get_action(np.asarray(states))
            actions = [(action, 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        
    env.close()


