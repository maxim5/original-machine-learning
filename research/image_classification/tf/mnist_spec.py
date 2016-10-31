#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np


def random_conv_layer(size, num, prob=0.8):
  if np.random.uniform() > prob:
    return [[size, 1, num], [1, size, num]]
  return [[size, size, num]]


activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']

hyper_params_generator = lambda: {
  'init_stdev': np.random.uniform(0.04, 0.06),

  'augment': {
    #'scale': [min(np.random.uniform(0.9, 1.1), 1.0),
    #          max(np.random.uniform(0.9, 1.1), 1.0)],
    'rotation_angle': np.random.choice([0, np.random.uniform(0, 20)]),
    'blur_sigma': np.random.choice([0, 10**np.random.uniform(-2, 0.2)]),
    'crop_size': np.random.choice(range(3)),
  },

  'optimizer': {
    'learning_rate': 10 ** np.random.uniform(-2, -4),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_conv_layer(size=np.random.choice([3, 5, 7]), num=np.random.choice([ 24,  32,  36])),
      'pools': [2, 2],
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.85, 1.0),
    },
    2: {
      'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([ 64,  96, 128])),
      'pools': [2, 2],
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.8, 1.0),
    },
    3: {
      'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([128, 256, 512])),
      'pools': [2, 2],
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.6, 1.0),
    }
  },

  'fc': {
    'size': np.random.choice([512, 768, 1024]),
    'activation': np.random.choice(activations),
    'dropout': np.random.uniform(0.5, 1.0),
  },
}
