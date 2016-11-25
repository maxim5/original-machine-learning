#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import datetime

from tflearn.datasets import cifar10

from image_classification.tf.conv_model import ConvModel
from image_classification.tf.hyper_tuner import HyperTuner
from image_classification.tf.tensorflow_impl import TensorflowRunner, TensorflowSolver
from image_classification.tf.data_set import Data, DataSet
from image_classification.tf.util import *

from cifar_spec import hyper_params_spec


def get_cifar10_data(validation_size=5000):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data('../../../dat/cifar-10-tf', one_hot=True)

  x_val = x_train[:validation_size]
  y_val = y_train[:validation_size]
  x_train = x_train[validation_size:]
  y_train = y_train[validation_size:]

  return Data(train=DataSet(x_train, y_train),
              validation=DataSet(x_val, y_val),
              test=DataSet(x_test, y_test))


# # {'conv': {1: {'activation': 'relu', 'dropout': 0.973845, 'filters': [[6, 6, 25]], 'pools': [2, 2]}, 2: {'activation': 'relu', 'dropout': 0.993991, 'filters': [[6, 6, 99]], 'pools': [2, 2]}, 3: {'activation': 'leaky_relu', 'dropout': 0.744906, 'filters': [[3, 3, 234]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'elu', 'dropout': 0.788669, 'size': 355}, 'init_stdev': 0.096934, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.000959}}
def stage1():
  data = get_cifar10_data()

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 256,
      'eval_batch_size': 2500,
      'epochs': 10,
      'dynamic_epochs': lambda acc: 3 if acc < 0.5 else 7 if acc < 0.6 else 10 if acc < 0.7 else 15 if acc < 0.75 else 20,
      'evaluate_test': True,
      'save_dir': '_models/cifar10/model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'save_accuracy_limit': 0.75,
    }

    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=data, runner=runner, **solver_params)
    return solver

  strategy_params = {
    'strategy': 'portfolio',
    'methods': ['ucb', 'pi', 'rand'],
    'probabilities': [0.3, 0.4, 0.3],
    'io_load_dir': '_models/cifar10/hyper/stage1-1.1',
    'io_save_dir': '_models/cifar10/hyper/stage1-1.1',
  }

  tuner = HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
  tuner.tune()


if __name__ == "__main__":
  stage1()
