#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:34:56 2019

@author: clytie
"""

import pickle
import subprocess
import sys
import numpy as np
import cv2


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rb") as f:
        frames = pickle.load(f)

    subprocess.Popen("mkdir tmp", shell=True)
    i = 0
    for frame in frames:
        cv2.imwrite(f"tmp/image{i}.png", frame[:, :, [2, 1, 0]])
        i += 1
    subprocess.Popen(f"ffmpeg -f image2 -i tmp/image%d.png -s 720x960 {i}.mp4", shell=True)