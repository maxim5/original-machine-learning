#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


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
  'init_stdev': spec.uniform(0.04, 0.06),

  'augment': {
     'crop_size': spec.choice(range(2)),
  },

  'optimizer': {
    'learning_rate': 10**spec.uniform(-2.5, -3.5),
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
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': random_conv_layer(size=spec.choice(range(3, 8)), num=spec.choice(range(64, 139))),
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

def uniform_snap(start, end):
  node = spec.uniform(start, end)

  def snap_to_ends(value):
    delta = (end - start) / 10.0
    if abs(value - start) < delta: return start
    if abs(value - end) < delta: return end
    return value

  return spec.wrap(node, snap_to_ends)

augment_spec = {
  'scale': [uniform_snap(0.75, 1.0), uniform_snap(1.0, 1.6)],
  'swirl': [uniform_snap(0, 1), uniform_snap(1, 50)],
  'rotation_angle': uniform_snap(0, 15),
  'blur_sigma': spec.uniform(0, 1),
  'crop_size': spec.choice(range(1, 4)),
}
