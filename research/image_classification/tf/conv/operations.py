#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import tensorflow as tf
from tensorflow.python.training import moving_averages

import variable

def leaky_relu(x, alpha=0.1):
  x = tf.nn.relu(x)
  m_x = tf.nn.relu(-x)
  x -= alpha * m_x
  return x

def prelu(x):
  shape = x.get_shape()
  with variable.scope('prelu'):
    alpha = variable.new(value=0, shape=shape[1:], name='alpha')
  x = tf.nn.relu(x) + tf.mul(alpha, (x - tf.abs(x))) * 0.5
  return x

ACTIVATIONS = {'leaky_relu': leaky_relu, 'prelu': prelu}
ACTIVATIONS.update({name: getattr(tf, name) for name in ['tanh']})
ACTIVATIONS.update({name: getattr(tf.nn, name) for name in ['relu', 'relu6', 'elu', 'sigmoid']})


def batch_normalization(incoming, is_training, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9):
  shape = incoming.get_shape()
  dimensions_num = len(shape)
  axis = list(range(dimensions_num - 1))

  with variable.scope('batchnorm'):
    beta = variable.new(value=beta, shape=[shape[-1]], name='beta')
    gamma = variable.new(value=gamma, shape=[shape[-1]], name='gamma')

    moving_mean = variable.new(value=0, shape=shape[-1:], trainable=False, name='moving_mean')
    moving_variance = variable.new(value=0, shape=shape[-1:], trainable=False, name='moving_variance')

  def update_mean_var():
    mean, variance = tf.nn.moments(incoming, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
      return tf.identity(mean), tf.identity(variance)

  mean, var = tf.cond(is_training, update_mean_var, lambda: (moving_mean, moving_variance))
  inference = tf.nn.batch_normalization(incoming, mean, var, beta, gamma, epsilon)
  inference.set_shape(shape)
  return inference

def dropout(incoming, is_training, keep_prob):
  return tf.cond(is_training, lambda: tf.nn.dropout(incoming, keep_prob), lambda: incoming)
