#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:33:05 2019

@author: clytie
"""

import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.logger import Logger
from raiden2 import Raiden2


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('explore_steps', 1024, 'number of steps to collect trajs')
flags.DEFINE_integer('total_updates', 1000, 'number of updates')
flags.DEFINE_integer('frame_stack', 4, 'number of frame')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_string('algorithm', 'basic_ppo', 'ppo algorithm')
flags.DEFINE_float('temperature', 1.0, 'random ratio')
flags.DEFINE_integer('train_epoch', 5, 'number of train epoch')


def main(_):
    explore_steps = FLAGS.explore_steps
    total_updates = FLAGS.total_updates
    frame_stack = FLAGS.frame_stack
    batch_size = FLAGS.batch_size
    algorithm = FLAGS.algorithm
    temperature = FLAGS.temperature
    train_epoch = FLAGS.train_epoch
    
    log_file = f'{algorithm}.log'
    save_path = f'{algorithm}_log'
    save_model_freq = 30
    action_space = 4
    size = 84
    death_frame = 29
    state_space = (size, size, frame_stack)
    
    logger = Logger(log_file, level='info')
    
    env = Raiden2(6666, num_envs=8, with_stack=True, num_frames=frame_stack)
    env_ids, states, rewards, dones = env.start()
    time.sleep(5)
    
    logger.logger.info(f'action_space={action_space}, state_space={state_space}')
    logger.logger.info(', '.join([f'{flag}={FLAGS.flag_values_dict()[flag]}' for flag in FLAGS.flag_values_dict()]))
    if algorithm == 'tsarppo':
        from algorithms.tsarppo import TSARPPO
        ppo = TSARPPO(action_space, state_space,
                      train_epoch=train_epoch,
                      temperature=temperature,
                      batch_size=batch_size,
                      save_path=save_path)
    elif algorithm == 'tsappo':
        from algorithms.tsappo import TSAPPO
        ppo = TSAPPO(action_space, state_space,
                     train_epoch=train_epoch,
                     temperature=temperature,
                     batch_size=batch_size,
                     save_path=save_path)
    elif algorithm == 'sappo':
        from algorithms.sappo import SAPPO
        ppo = SAPPO(action_space, state_space,
                    train_epoch=train_epoch,
                    temperature=temperature,
                    batch_size=batch_size,
                    save_path=save_path)
    elif algorithm == 'sappov2':
        from algorithms.sappo_v2 import SAPPOV2
        ppo = SAPPOV2(action_space, state_space,
                      train_epoch=train_epoch,
                      temperature=temperature,
                      batch_size=batch_size,
                      save_path=save_path)
    else:
        from algorithms.basic_ppo import BasicPPO
        ppo = BasicPPO(action_space, state_space,
                       train_epoch=train_epoch,
                       temperature=temperature,
                       batch_size=batch_size,
                       save_path=save_path)
    
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
                logger.logger.info("no enough data, continue collect data.")
                continue

        logger.logger.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        ppo.update(
            s_batchs, a_batchs, r_batchs, d_batchs,
            min(0.9, nth_trajectory / total_updates)
            )
        ppo.sw.add_scalar(
            'reward_mean',
            env.mean_reward,
            global_step=nth_trajectory
            )

        if nth_trajectory % save_model_freq == 0:
            ppo.save_model()
    env.close()


if __name__ == '__main__':
    tf.app.run()

