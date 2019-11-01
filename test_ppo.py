#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:26:45 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    import logging
    import tensorflow as tf
    
    from tqdm import tqdm
    from algorithms.ppo import PPO
    from raiden2 import Raiden2
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    explore_steps = 1024
    total_updates = 1000
    save_model_freq = 30
    action_space = 4
    frame_stack = 4
    size = 84
    death_frame = 29
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
    
    ppo = PPO(action_space, obs_fn, model_fn, train_epoch=5, batch_size=64, save_path='./raiden2_model')
    
    env = Raiden2(6666, num_envs=1, with_stack=True)
    env_ids, states, rewards, dones = env.start()
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = ppo.get_action(np.asarray(states))
            actions = [(action, 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
    env.close()
