#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:23:36 2019

@author: clytie
"""

import tensorflow as tf
from algorithms.basic_ppo import BasicPPO


class SAPPO(BasicPPO):
    def __init__(self,
                 n_action, dim_ob_image,
                 rnd=1,
                 temperature=1.0,
                 discount=0.99,
                 gae=0.95,
                 vf_coef=1.0,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 train_epoch=10,
                 batch_size=64,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 clip_schedule=lambda x: max(0.2, (1 - x)) * 0.5,
                 save_path="./sappo_log"):
        super().__init__(
                n_action=n_action,
                dim_ob_image=dim_ob_image,
                rnd=rnd,
                temperature=temperature,
                discount=discount,
                gae=gae,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                train_epoch=train_epoch,
                batch_size=batch_size,
                lr_schedule=lr_schedule,
                clip_schedule=clip_schedule,
                save_path=save_path
            )
        
    def _build_network(self):
        self.ob_image = tf.placeholder(
            tf.float32, [None, *self.dim_ob_image], name="image_observation")
        batch_size = tf.shape(self.ob_image)[0]
        
        self.conv1 = tf.layers.conv2d(self.ob_image, 32, 8, 4, activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu)
        
        # spatial part
        H = 512
        L = 7 * 7
        D = 64
        
        def _project_features(feature, D, L):
            with tf.variable_scope('project_features'):
                w = tf.get_variable(
                        'w', [D, D], 
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                feature_flatten = tf.reshape(feature, [-1, D])
                feature_project = tf.matmul(feature_flatten, w)  
                feature_project = tf.reshape(feature_project, [-1, L, D])
                return feature_project
            
        def _spatial_attention(feature, feature_project, H, D, L):
            with tf.variable_scope('spatial_attention'):
                w = tf.get_variable(
                        'w', [D, 1],
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                b = tf.get_variable(
                        'b', [D], initializer=tf.constant_initializer(0.0))
    
                h_att = tf.nn.relu(feature_project + b)
                out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, D]), w), [-1, L])
                alpha = tf.nn.softmax(out_att)
                context = feature * tf.expand_dims(alpha + 1, 2)
                return context, alpha
        
        # Spatial Attention: The Generalization of Global Average Pooling
        with tf.variable_scope('spatial'):
            feature = tf.reshape(self.conv3, [batch_size, L, D])
            feature_project = _project_features(feature=feature,
                                                D=D,
                                                L=L)
            context, alpha = _spatial_attention(feature=feature,
                                                feature_project=feature_project,
                                                H=H,
                                                D=D,
                                                L=L)
            self.spatial_alpha = alpha
            self.spatial_attention = context
        
        inputs = tf.contrib.layers.flatten(self.spatial_attention)
        inputs = tf.layers.dense(inputs, H, activation=tf.nn.relu)
        self.logit_action_probability = tf.layers.dense(
            inputs, self.n_action,
            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            )
        self.state_value = tf.squeeze(tf.layers.dense(
            inputs, 1, kernel_initializer=tf.truncated_normal_initializer())
            )

