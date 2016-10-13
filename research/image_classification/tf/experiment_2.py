#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from common import *
from conv_model import ConvModel


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


def save_hyper(accuracy, hyper_params, path='best-hyper-%.4f.txt', limit=0.992):
  if accuracy < limit:
    return

  filename = path % limit
  with open(filename, 'a') as file_:
    file_.write('accuracy=%.4f -> %s\n' % (accuracy, dict_to_str(hyper_params)))


def hyper_tune(data_sets, model):
  best_accuracy = 0
  while True:
    hyper_params = default_hyper_params.copy()
    hyper_params['epochs'] = 10
    tuned_params = {
      'init_stdev': np.random.uniform(0.04, 0.06),
      'learning_rate': 10**np.random.uniform(-2, -4),

      'conv_filters': random_conv_spec(),
      'conv_activation': np.random.choice(['relu', 'elu']),
      'conv_dropout': np.random.uniform(0.7, 1.0),

      'fc_size': np.random.choice([512, 768, 1024, 2048]),
      'fc_activation': np.random.choice(['relu', 'elu']),
      'fc_dropout': np.random.uniform(0.5, 1.0),
    }
    hyper_params.update(tuned_params)

    tf.reset_default_graph()
    accuracy = train(data_sets=data_sets, model=model, **hyper_params)

    marker = '   '
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      marker = '!!!'
    log('%s accuracy=%.4f, tuned_params: %s' % (marker, accuracy, dict_to_str(tuned_params)))

    save_hyper(accuracy, hyper_params)


def train_best_candidate(data_sets, model):
  hyper_params = default_hyper_params.copy()
  hyper_params.update({'batch_size': 128, 'conv_activation': 'elu', 'conv_dropout': 0.79266, 'conv_filters': [[[5, 5, 24]], [[3, 3, 64]], [[5, 5, 256]]], 'conv_pools': [[2, 2]], 'epochs': 10, 'fc_activation': 'relu', 'fc_dropout': 0.85501, 'fc_size': 1024, 'init_stdev': 0.05557, 'learning_rate': 0.00108})
  hyper_params['epochs'] = 30
  train(data_sets=data_sets, model=model, **hyper_params)


def train_default(data_sets, model):
  train(data_sets=data_sets, model=model, **default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  hyper_tune(mnist, conv_model)
