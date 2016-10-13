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
    [[3, 3, 256]],
  ],
  'conv_pools': [
    [2, 2]
  ],
  'fc_size': 1024,
  'dropout_conv': 0.8,
  'dropout_fc': 0.5,
}


def save_hyper(accuracy, hyper_params, path='best-hyper-%.4f.txt', limit=0.992):
  if accuracy < limit:
    return

  filename = path % limit
  with open(filename, 'a') as file_:
    file_.write('accuracy=%.4f -> %s' % (accuracy, str(hyper_params)))


def hyper_tune(data_sets, model):
  best_accuracy = 0
  while True:
    hyper_params = default_hyper_params.copy()
    hyper_params['epochs'] = 12
    # accuracy=0.9618, tuned_params:  {'init_stdev': 0.05529920193609984}
    tuned_params = {
      'init_stdev': np.random.uniform(0.04, 0.06),
      'learning_rate': 10**np.random.uniform(-2, -4),
      'dropout_conv': np.random.uniform(0.5, 1.0),
      'dropout_fc': np.random.uniform(0.2, 1.0),
    }
    hyper_params.update(tuned_params)

    tf.reset_default_graph()
    accuracy = train(data_sets=data_sets, model=model, **hyper_params)

    marker = '   '
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      marker = '!!!'
    log("%s accuracy=%.4f, tuned_params: " %  (marker, accuracy), tuned_params)

    save_hyper(accuracy, hyper_params)


def train_best_candidate(data_sets, model):
  hyper_params = default_hyper_params.copy()
  hyper_params.update({'learning_rate': 0.0006568582429913265, 'dropout_fc': 0.8558553724757245, 'dropout_conv': 0.9400811432728358, 'init_stdev': 0.043707865797119355})
  hyper_params['epochs'] = 30
  train(data_sets=data_sets, model=model, **hyper_params)


def train_default(data_sets, model):
  train(data_sets=data_sets, model=model, **default_hyper_params)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  conv_model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  hyper_tune(mnist, conv_model)
