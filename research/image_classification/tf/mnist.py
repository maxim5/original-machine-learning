#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import copy
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from conv_model import ConvModel
from hyper import Solver, HyperTuner, HyperParamsFile


default_hyper_params = {
  'batch_size': 128,
  'epochs': 10,
  'init_stdev': 0.05,
  'learning_rate': 0.001,

  'conv': {
    'layers_num': 3,
    1: {
      'filters': [[3, 3,  36], [3, 3,  36]],
      'pools': [2, 2],
      'activation': 'elu',
      'dropout': 0.9,
    },
    2: {
      'filters': [[3, 3,  64], [5, 3,  64]],
      'pools': [2, 2],
      'activation': 'elu',
      'dropout': 0.8,
    },
    3: {
      'filters': [[3, 3, 256], [3, 3, 256]],
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


def random_conv_layer(size, num):
  return [[size, size, num]]


def hyper_tune(data_sets, model):
  fixed_params = copy.deepcopy(default_hyper_params)
  fixed_params['epochs'] = 10

  activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']
  tuned_params_generator = lambda : {
    'init_stdev': np.random.uniform(0.04, 0.06),
    'learning_rate': 10**np.random.uniform(-2, -4),

    'conv': {
      'layers_num': 3,
      1: {
        'filters': random_conv_layer(size=np.random.choice([3, 5, 7]), num=np.random.choice([ 24,  32,  36])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.7, 1.0),
      },
      2: {
        'filters': random_conv_layer(size=np.random.choice([3, 5, 7]), num=np.random.choice([ 64,  96, 128])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.7, 1.0),
      },
      3: {
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([128, 256, 512])),
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.7, 1.0),
      }
    },

    'fc': {
      'size': np.random.choice([512, 768, 1024, 2048]),
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.5, 1.0),
    },
  }

  solver = Solver(data_sets, model)
  tuner = HyperTuner(solver, save_path='best-hyper-%.4f.txt', save_limit=0.9930)
  tuner.tune(fixed_params, tuned_params_generator)


def train_best_candidate(data_sets, model, from_file='best-hyper-0.9920.txt', start=0, end=5, epochs=50):
  hyper_file = HyperParamsFile(from_file)
  hyper_list = hyper_file.get_all()

  solver = Solver(data_sets, model)

  for index, hyper_params in enumerate(hyper_list[start:end]):
    hyper_params['epochs'] = epochs
    max_val_accuracy, test_accuracy = solver.train(evaluate_test=True, **hyper_params)
    line = '# trained_epochs=%d validation_accuracy=%.4f test_accuracy=%.4f' % (epochs, max_val_accuracy, test_accuracy)
    hyper_file.update_pack(index, line)
    hyper_file.save_all()


def train_default(data_sets, model):
  solver = Solver(data_sets, model)
  solver.train(evaluate_test=True, **default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  hyper_tune(mnist, conv_model)
