#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:07:35 2019

@author: clytie
"""

import tensorflow as tf


writer = tf.summary.FileWriter(logdir='graph', graph=tf.get_default_graph())
writer.close()