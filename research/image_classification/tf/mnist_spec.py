#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from image_classification.tf import spec


def random_conv_layer(size, num, prob=0.8):
  def if_cond(switch, size, num):
    if switch > prob:
      return [[size, 1, num], [1, size, num]]
    return [[size, size, num]]

  return spec.merge(if_cond, spec.uniform(), size, num)


activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']

hyper_params = {
  'init_stdev': spec.uniform(0.04, 0.06),

  'augment': {
    'scale': [spec.min_(spec.uniform(0.9, 1.1), 1.0),
              spec.max_(spec.uniform(0.9, 1.1), 1.0)],
    'rotation_angle': spec.choice([0, spec.uniform(0, 20)]),
    'blur_sigma': spec.choice([0, 10**spec.uniform(-2, 0.2)]),
    'crop_size': spec.choice(range(3)),
  },

  'optimizer': {
    'learning_rate': 10**spec.uniform(-2, -4),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_conv_layer(size=spec.choice([3, 5, 7]), num=spec.choice([ 24,  32,  36])),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': random_conv_layer(size=spec.choice([3, 5,  ]), num=spec.choice([ 64,  96, 128])),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.8, 1.0),
    },
    3: {
      'filters': random_conv_layer(size=spec.choice([3, 5,  ]), num=spec.choice([128, 256, 512])),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.6, 1.0),
    }
  },

  'fc': {
    'size': spec.choice([512, 768, 1024]),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.5, 1.0),
  },
}
