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
  'init_stdev': 0.01,
  'learning_rate': 0.001,
  'conv_filters': [
    # [[7, 1,  32], [1, 7,  32]],
    # [[7, 1,  64], [1, 7,  64]],
    # [[3, 1, 96], [1, 3, 96]],

    [[3, 3, 32]],
    [[3, 3, 64]],
    [[3, 3, 128]],
  ],
  'conv_pools': [
    [1, 2, 2, 1]
  ],
  'fc_size': 1024,
  'dropout_conv': 0.8,
  'dropout_fc': 0.5,
}


def hyper_tune(data_sets, model):
  best_accuracy = 0
  while True:
    hyper_params = default_hyper_params.copy()
    hyper_params['init_stdev'] = 10**np.random.uniform(-1, -3)
    hyper_params['learning_rate'] = 10**np.random.uniform(-3, -4)
    hyper_params['dropout_conv'] = np.random.uniform(0.5, 1.0)
    hyper_params['dropout_fc'] = np.random.uniform(0.2, 1.0)
    hyper_params['epochs'] = 15

    tf.reset_default_graph()
    accuracy = train(data_sets=data_sets, model=model, **hyper_params)

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      log("!!! new best_acc=%.4f" %  best_accuracy)


def train_best_candidate(data_sets, model):
  hyper_params = default_hyper_params.copy()
  hyper_params.update({'epochs': 20, 'dropout_fc': 0.551, 'learning_rate': 0.00076, 'dropout_conv': 0.9409, 'init_stdev': 0.0248})
  train(data_sets=data_sets, model=model, **hyper_params)


def train_default(data_sets, model):
  train(data_sets=data_sets, model=model, **default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  hyper_tune(mnist, conv_model)
