#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime

import tflearn
from tflearn.datasets.mnist import read_data_sets

from conv_model import ConvModel
from hyper_tuner import *
from tensorflow_impl import *


def random_conv_layer(size, num, prob=0.8):
  if np.random.uniform() > prob:
    return [[size, 1, num], [1, size, num]]
  return [[size, size, num]]


def hyper_tune_ground_up():
  mnist = read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)

  activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']
  tuned_params_generator = lambda: {
    'init_stdev': np.random.uniform(0.04, 0.06),

    'augment': {
      'rotation_angle': np.random.uniform(0, 15),
      'blur_sigma': np.random.uniform(0, 5),
      'crop_size': np.random.choice(range(5)),
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
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([ 24,  32,  36])),
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

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 1024,
      'eval_batch_size': 5000,
      'epochs': 12,
      'dynamic_epochs': lambda acc: 3 if acc < 0.8000 else 5 if acc < 0.9800 else 10 if acc < 0.9920 else 15,
      'evaluate_test': True,
      'save_dir': 'model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'save_accuracy_limit': 0.9940
    }

    augment_params = hyper_params.get('augment')
    if augment_params:
      augmentation = tflearn.ImageAugmentation()
      rotation_angle = augment_params.get('rotation_angle')
      if rotation_angle:
        augmentation.add_random_rotation(max_angle=rotation_angle)
      blur_sigma = augment_params.get('blur_sigma')
      if blur_sigma:
        augmentation.add_random_blur(sigma_max=blur_sigma)
      crop_size = augment_params.get('crop_size')
      if crop_size:
        augmentation.add_random_crop(crop_shape=(28-crop_size, 28-crop_size), padding=(crop_size, crop_size))
    else:
      augmentation = None

    runner = TensorflowRunner(model=conv_model, **hyper_params)
    solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, **solver_params)
    return solver

  tuner = HyperTuner()
  tuner.tune(solver_generator, tuned_params_generator)


def fine_tune(only_test=False):
  mnist = read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)

  model_path = 'model-zoo/2016-10-21-BT5CES'
  solver_params = {
    'batch_size': 1024,
    'eval_batch_size': 5000,
    'epochs': 0 if only_test else 50,
    'evaluate_test': True,
    'save_dir': model_path,
    'load_dir': model_path,
  }

  runner = TensorflowRunner(model=conv_model)
  solver = TensorflowSolver(data=mnist, runner=runner, **solver_params)
  solver.train()


if __name__ == "__main__":
  hyper_tune_ground_up()
