#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from conv_model import ConvModel
from hyper import Solver, HyperTuner


default_hyper_params = {
  'batch_size': 128,
  'epochs': 10,
  'init_stdev': 0.05,
  'learning_rate': 0.001,

  'conv_filters': [
    [[1, 5,  36], [5, 1,  36]],
    [[1, 5,  64], [5, 1,  64]],
    [[1, 5, 256], [5, 1, 256]],
  ],
  'conv_pools': [
    [2, 2]
  ],
  'conv_activation': 'elu',
  'conv_dropout': 0.8,

  'fc_activation': 'elu',
  'fc_size': 1024,
  'fc_dropout': 0.5,
}


def random_conv_spec():
  size1 = np.random.choice([3, 5, 7])
  size2 = np.random.choice([3, 5, 7])
  size3 = np.random.choice([3, 5])

  num1 = np.random.choice([24, 32, 36])
  num2 = np.random.choice([64, 96, 128])
  num3 = np.random.choice([128, 256, 512])

  spec = [
    [[size1, size1, num1]],
    [[size2, size2, num2]],
    [[size3, size3, num3]] if num3 > 0 else None,
  ]
  return [x for x in spec if x is not None]


def hyper_tune(data_sets, model):
  fixed_params = default_hyper_params.copy()
  fixed_params['epochs'] = 10
  tuned_params_generator = lambda : {
    'init_stdev': np.random.uniform(0.04, 0.06),
    'learning_rate': 10**np.random.uniform(-2, -4),

    'conv_filters': random_conv_spec(),
    'conv_activation': np.random.choice(['relu', 'elu']),
    'conv_dropout': np.random.uniform(0.7, 1.0),

    'fc_size': np.random.choice([512, 768, 1024, 2048]),
    'fc_activation': np.random.choice(['relu', 'elu']),
    'fc_dropout': np.random.uniform(0.5, 1.0),
  }

  solver = Solver(data_sets, model)
  tuner = HyperTuner(solver, save_path='best-hyper-%.4f.txt')
  tuner.tune(fixed_params, tuned_params_generator)


def train_best_candidate(data_sets, model):
  hyper_params = default_hyper_params.copy()
  hyper_params.update({'batch_size': 128, 'conv_activation': 'relu', 'conv_dropout': 0.76786, 'conv_filters': [[[3, 3, 32]], [[5, 5, 64]], [[3, 3, 512]]], 'conv_pools': [[2, 2]], 'epochs': 10, 'fc_activation': 'elu', 'fc_dropout': 0.99931, 'fc_size': 768, 'init_stdev': 0.04807, 'learning_rate': 0.00235})
  hyper_params['epochs'] = 30
  solver = Solver(data_sets, model)
  solver.train(**hyper_params)


def train_default(data_sets, model):
  solver = Solver(data_sets, model)
  solver.train(**default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  hyper_tune(mnist, conv_model)
