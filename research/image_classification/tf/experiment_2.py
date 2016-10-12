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
  'learning_rate': 0.001,
  'conv_filters': [
    [[3, 1,  32], [1, 3,  32]],
    [[3, 1,  64], [1, 3,  64]],
    [[3, 1, 128], [1, 3, 128]],
  ],
  'conv_pools': [
    [1, 2, 2, 1]
  ],
  'dropout_conv': 0.95,
  'dropout_fc': 0.9,
}


def hyper_tune(data_sets, model):
  best_accuracy = 0
  while True:
    tf.reset_default_graph()

    hyper_params = default_hyper_params.copy()
    hyper_params['learning_rate'] = 10**np.random.uniform(-3, -5)
    hyper_params['dropout_conv'] = np.random.uniform(0.8, 1.0)
    hyper_params['dropout_fc'] = np.random.uniform(0.8, 1.0)

    accuracy = train(data_sets=data_sets, model=model, **hyper_params)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      log("!!! new best_acc=%.4f" %  best_accuracy)


# def run(data_sets):
#   model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
#   hyper = {'epochs': 20, 'learning_rate': 0.000849798099730975, 'dropout_fc': 0.8962278592597537, 'dropout_conv': 0.9673306165168953, 'batch_size': 128}
#   train(data_sets=data_sets, model=model, **hyper)


def experiment(data_sets, model):
  train(data_sets=data_sets, model=model, **default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  experiment(mnist, conv_model)
