#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:44:42 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    import cv2
    import logging
    import tensorflow as tf
    
    from tqdm import tqdm
    from collections import deque, defaultdict
    from algorithms.ppo import PPO
    from raiden2 import Raiden2
    from functools import partial
    from skimage.color import rgb2gray
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    explore_steps = 1024
    total_updates = 2000
    save_model_freq = 100
    action_space = 16
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
    
    ppo = PPO(action_space, obs_fn, model_fn, train_epoch=5, batch_size=32)
    
    env = Raiden2(6666, num_envs=8, with_stack=False)
    env_ids, states, rewards, dones = env.start()
    env_states = defaultdict(partial(deque, maxlen=frame_stack))
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            sts = []
            for env_id, state in zip(env_ids, states):
                st = np.zeros((size, size, frame_stack), dtype=np.float32)
                im = cv2.resize(rgb2gray(state), (size, size))
                env_states[env_id].append(im)
                while len(env_states[env_id]) < frame_stack:
                    env_states[env_id].append(im)
                for idx, each in enumerate(env_states[env_id]):
                    st[:, :, idx: idx + 1] = each[:, :, np.newaxis]
                sts.append(st)
            actions = ppo.get_action(np.asarray(sts))
            actions = [(action // 4, action % 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        s_batchs, a_batchs, r_batchs, d_batchs = env.get_episodes()
        s_batchs = [[cv2.resize(rgb2gray(s), (size, size)) for s in s_batch] for s_batch in s_batchs]
        stack_s_batchs = []
        for s_batch in s_batchs:
            stack_s_batch = []
            stack = deque(maxlen=frame_stack)
            st = np.zeros((size, size, frame_stack), dtype=np.float32)
            for s in s_batch:
                stack.append(s)
                while len(stack) < frame_stack:
                    stack.append(s)
                for idx, each in enumerate(stack):
                    st[:, :, idx: idx + 1] = each[:, :, np.newaxis]
                stack_s_batch.append(st.copy())
            stack_s_batchs.append(stack_s_batch)
        a_batchs = [[a[0] * 4 + a[1] for a in a_batch] for a_batch in a_batchs]
        r_batchs = [[sum(r) for r in r_batch] for r_batch in r_batchs]
        
        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        ppo.update(stack_s_batchs, a_batchs, r_batchs, d_batchs,
                       min(0.9, nth_trajectory / total_updates))
        ppo.sw.add_scalar(
                    'reward_mean',
                    env.mean_reward,
                    global_step=nth_trajectory)

        if nth_trajectory % save_model_freq == 0:
            ppo.save_model()
    env.close()

