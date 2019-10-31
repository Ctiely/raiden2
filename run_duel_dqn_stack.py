#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:45:29 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    from raiden2 import Raiden2
    from algorithms.replay_buffer import ReplayBuffer
    from algorithms.duel_dqn import DuelDQN
    from utils.logger import Logger
    
    
    logger = Logger('duel_dqn.log', level='info')
    action_space = 16
    state_space = (84, 84, 4)
    
    memory = ReplayBuffer(max_size=250000)
    env = Raiden2(6666, num_envs=8, with_stack=True)
    env_ids, states, rewards, dones = env.start()
    print("pre-train: ")
    for _ in tqdm(range(5000)):
        actions = np.random.randint(action_space, size=env.num_srd)
        actions = [(action // 4, action % 4) for action in actions]
        env_ids, states, rewards, dones = env.step(env_ids, actions)
    s_batch, a_batch, r_batch, d_batch = env.get_episodes()
    a_batch = [[a[0] * 4 + a[1] for a in a_bth] for a_bth in a_batch]
    r_batch = [[sum(r) for r in r_bth] for r_bth in r_batch]

    memory.add(s_batch, a_batch, r_batch, d_batch)
    DuelDQNetwork = DuelDQN(action_space, state_space, 
                            epsilon_schedule=lambda x: max(0.1, (1e6-x) / 1e6),
                            save_path="./duel_dqn_log")
    
    print("start train: ")
    for step in range(1000000):
        for _ in tqdm(range(64)):
            actions = DuelDQNetwork.get_action(np.asarray(states))
            actions = [(action // 4, action % 4) for action in actions]
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        if step % 10 == 0:
            DuelDQNetwork.sw.add_scalar(
                    'reward_mean',
                    env.mean_reward,
                    global_step=step // 10)
            logger.logger.info(f'>>>>{env.mean_reward}, nth_step{step}, buffer{len(memory)}')
        s_batch, a_batch, r_batch, d_batch = env.get_episodes()
        a_batch = [[a[0] * 4 + a[1] for a in a_bth] for a_bth in a_batch]
        r_batch = [[sum(r) for r in r_bth] for r_bth in r_batch]
        memory.add(s_batch, a_batch, r_batch, d_batch)
        for _ in tqdm(range(10)):
            batch_samples = memory.sample(32)
            DuelDQNetwork.update(batch_samples, sw_dir="duel_dqn")

    env.close()

