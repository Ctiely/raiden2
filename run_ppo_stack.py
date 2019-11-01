#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:11:53 2019

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
    
    ppo = PPO(action_space, obs_fn, model_fn, train_epoch=5, batch_size=64, save_path='./ppo_log_oneplayer_stack')
    
    env = Raiden2(6666, num_envs=8, with_stack=True)
    env_ids, states, rewards, dones = env.start()
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = ppo.get_action(np.asarray(states))
            actions = [(action, 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        s_batchs, a_batchs, r_batchs, d_batchs = env.get_episodes()
        a_batchs = [[a[0] for a in a_batch] for a_batch in a_batchs]
        r_batchs = [[r[0] for r in r_batch] for r_batch in r_batchs]
        if death_frame > 0:
            batch_size = len(d_batchs)
            retain_s_batch = []
            retain_a_batch = []
            retain_r_batch = []
            retain_d_batch = []
            for i in range(batch_size):
                traj_size = len(d_batchs[i])
                if traj_size >= death_frame:
                    if d_batchs[i][-1] == True:
                        r_episode, d_episode = r_batchs[i], d_batchs[i]
                        r_episode[-death_frame] = r_episode[-1]
                        d_episode[-death_frame] = True
                    
                    retain_s_batch.append(s_batchs[i][:-death_frame + 1])
                    retain_a_batch.append(a_batchs[i][:-death_frame + 1])
                    retain_r_batch.append(r_batchs[i][:-death_frame + 1])
                    retain_d_batch.append(d_batchs[i][:-death_frame + 1])
            s_batchs = retain_s_batch
            a_batchs = retain_a_batch
            r_batchs = retain_r_batch
            d_batchs = retain_d_batch
            if not len(s_batchs):
                nth_trajectory -= 1
                logging.info("no enough data, continue collect data.")
                continue

        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        ppo.update(s_batchs, a_batchs, r_batchs, d_batchs,
                       min(0.9, nth_trajectory / total_updates))
        ppo.sw.add_scalar(
                    'reward_mean',
                    env.mean_reward,
                    global_step=nth_trajectory)

        if nth_trajectory % save_model_freq == 0:
            ppo.save_model()
    env.close()

