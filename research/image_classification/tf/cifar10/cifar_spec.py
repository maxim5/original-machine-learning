#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from image_classification.tf import spec

def random_conv_layer(size, num):
  return [[size, size, num]]

def random_conv_layer_plus(size, num, prob=0.8):
  def if_cond(switch, size, num):
    if switch > prob:
      return [[size, 1, num], [1, size, num]]
    return [[size, size, num]]

  return spec.merge([spec.uniform(), size, num], if_cond)

activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']

hyper_params_spec = {
  'init_stdev': 10**spec.uniform(-2, 0),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3, -1),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_conv_layer(size=spec.choice(range(3, 8)), num=spec.choice(range(24, 41))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.8, 1.0),
    },
    2: {
      'filters': random_conv_layer(size=spec.choice(range(3, 8)), num=spec.choice(range(64, 141))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.8, 1.0),
    },
    3: {
      'filters': random_conv_layer(size=spec.choice(range(3, 6)), num=spec.choice(range(128, 513))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.6, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(512, 1025)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.5, 1.0),
  },
}
