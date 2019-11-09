#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:11:32 2019

@author: clytie
"""

import tensorflow as tf
from algorithms.basic_ppo import BasicPPO


class TSAPPO(BasicPPO):
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
                 save_path="./tsappo_log"):
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
        self.H = 512
        
        self.obs_permute = tf.transpose(self.ob_image, perm=[0, 3, 1, 2])
        self.obs_reshape = tf.reshape(self.obs_permute, [-1, 84, 84, 1])
        
        self.conv1 = tf.layers.conv2d(self.obs_reshape, 32, 8, 4, activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu)
        
        # spatial part
        self.T = self.dim_ob_image[-1]
        self.L = 7 * 7
        self.D = 64
        
        def _project_features(feature, D, L, reuse):
            with tf.variable_scope('project_features', reuse=reuse):
                w = tf.get_variable(
                        'w', [D, D], 
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                feature_flatten = tf.reshape(feature, [-1, D])
                feature_project = tf.matmul(feature_flatten, w)  
                feature_project = tf.reshape(feature_project, [-1, L, D])
                return feature_project
            
        def _spatial_attention(feature, feature_project, D, L, reuse):
            with tf.variable_scope('spatial_attention', reuse=reuse):
                w = tf.get_variable(
                        'w', [D, 1],
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                b = tf.get_variable(
                        'b', [D], initializer=tf.constant_initializer(0.0))
    
                h_att = tf.nn.relu(feature_project + b)
                out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, D]), w), [-1, L])
                alpha = tf.nn.softmax(out_att)
                context = tf.reduce_sum(feature * tf.expand_dims(alpha, 2), 1, name='context')
                return context, alpha
        
        with tf.variable_scope('spatial'):
            self.feature = tf.reshape(self.conv3, [batch_size, self.T, self.L, self.D])
            self.features = tf.split(self.feature, self.T, axis=1)
            self.spatial_alphas = []
            self.spatial_contexts = []
            
            for t in range(self.T):
                reuse = (t != 0)
                feature = tf.squeeze(self.features[t], [1])
                feature_project = _project_features(feature=feature,
                                                    D=self.D,
                                                    L=self.L,
                                                    reuse=reuse)
                context, alpha = _spatial_attention(feature=feature,
                                                    feature_project=feature_project,
                                                    D=self.D,
                                                    L=self.L,
                                                    reuse=reuse)
                self.spatial_alphas.append(alpha)
                self.spatial_contexts.append(context)
            self.spatial_context = tf.concat(self.spatial_contexts, axis=1)
            self.spatial_output = tf.layers.dense(
                    self.spatial_context, self.H, activation=tf.nn.relu
                    )

        # temporal part
        with tf.variable_scope('temporal'):
            self.flatten = tf.contrib.layers.flatten(self.conv3)
            self.dense = tf.layers.dense(self.flatten, self.H, activation=tf.nn.relu)
            self.temporal_context = tf.reshape(self.dense, [batch_size, self.T, self.H])
            
            with tf.variable_scope('temporal_attention'):
                attention_v = tf.get_variable(
                        'v', [self.H, 1], 
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                attention_va = tf.nn.tanh(
                        tf.map_fn(lambda x: tf.matmul(x, attention_v), self.temporal_context)
                        )
                self.temporal_alpha = tf.nn.softmax(attention_va, axis=1)
                self.temporal_output = tf.reduce_sum(
                        tf.multiply(self.temporal_context, self.temporal_alpha), axis=1
                        )
        
        inputs = tf.concat([self.spatial_output, self.temporal_output], axis=1)
        inputs = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
        self.logit_action_probability = tf.layers.dense(
            inputs, self.n_action,
            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            )
        self.state_value = tf.squeeze(tf.layers.dense(
            inputs, 1, kernel_initializer=tf.truncated_normal_initializer())
            )

