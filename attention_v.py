#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:46:40 2019

@author: clytie
"""

import numpy as np
import skimage
import cv2
import pickle
import time
import matplotlib.pyplot as plt

from algorithms.tsarppo import TSAPPO


frames = pickle.load(open('frames.pkl', 'rb'))
states = pickle.load(open('states.pkl', 'rb'))

ppo = TSAPPO(4, (84, 84, 8), save_path='tsappo')


for index in range(len(frames)):
    print(index)
    time.sleep(1)
    alphas = ppo.sess.run(ppo.spatial_alphas, feed_dict={ppo.ob_image: np.asarray([states[index]])})
    image = cv2.resize(frames[index], (224, 224))
    alpha_img = skimage.transform.pyramid_expand(alphas[-1].reshape(7,7), upscale=32, sigma=20)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.dstack(((np.array(image) / 255), 1 - alpha_img / np.max(alpha_img))))
    fig.savefig(f'attention{index}.png')

    
