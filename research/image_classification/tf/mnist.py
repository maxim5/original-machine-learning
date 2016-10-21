#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime
import numpy as np
from tflearn.datasets.mnist import read_data_sets

from conv_model import ConvModel
from hyper_tuner import *
from tensorflow_impl import *


default_hyper_params = {
  'init_stdev': 0.05,

  'adam': {
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': [[3, 3,  32]],
      'pools': [2, 2],
      'activation': 'elu',
      'dropout': 0.9,
    },
    2: {
      'filters': [[3, 3,  64]],
      'pools': [2, 2],
      'activation': 'elu',
      'dropout': 0.8,
    },
    3: {
      'filters': [[3, 3, 256]],
      'pools': [2, 2],
      'activation': 'elu',
      'dropout': 0.7,
    },
  },

  'fc': {
    'size': 1024,
    'activation': 'elu',
    'dropout': 0.5,
  }
}


def random_conv_layer(size, num, prob=0.8):
  if np.random.uniform() > prob:
    return [[size, 1, num], [1, size, num]]
  return [[size, size, num]]


def hyper_tune_ground_up():
  mnist = read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)

  fixed_params = copy.deepcopy(default_hyper_params)
  activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']
  tuned_params_generator = lambda: {
    'init_stdev': np.random.uniform(0.04, 0.06),

    'adam': {
      'learning_rate': 10 ** np.random.uniform(-2, -4),
    },

    'conv': {
      'layers_num': 3,
      1: {
        'filters': random_conv_layer(size=np.random.choice([3, 5, 7]), num=np.random.choice([ 24,  32,  36,  48])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.9, 1.0),
      },
      2: {
        'filters': random_conv_layer(size=np.random.choice([3, 5, 7]), num=np.random.choice([ 64,  96, 128, 192])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.8, 1.0),
      },
      3: {
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([128, 256, 512, 768])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.6, 1.0),
      }
    },

    'fc': {
      'size': np.random.choice([512, 768, 1024, 1280]),
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.5, 1.0),
    },
  }

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 256,
      'epochs': 10,
      'dynamic_epochs': lambda acc: 3 if acc < 0.8 else 5 if acc < 0.99 else 10 if acc < 0.994 else 15,
      'evaluate_test': False,
      'save_dir': 'model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'save_accuracy_limit': 0.9940
    }

    runner = TensorflowRunner(model=conv_model, **hyper_params)
    solver = TensorflowSolver(data=mnist, runner=runner, **solver_params)
    return solver

  tuner = HyperTuner()
  tuner.tune(solver_generator, fixed_params, tuned_params_generator)


def fine_tune(eval_test=False):
  mnist = read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)

  model_path = 'model-zoo/2016-10-21-BT5CES'
  solver_params = {
    'batch_size': 1024,
    'epochs': 0 if eval_test else 50,
    'evaluate_test': eval_test,
    'eval_test_batch_size': 5000,
    'save_dir': model_path,
    'load_dir': model_path,
  }

  runner = TensorflowRunner(model=conv_model)
  solver = TensorflowSolver(data=mnist, runner=runner, **solver_params)
  solver.train()


if __name__ == "__main__":
  fine_tune(eval_test=True)
