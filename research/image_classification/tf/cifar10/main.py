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
from image_classification.tf.curve_predictor import LinearCurvePredictor

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


# {'conv': {1: {'activation': 'relu', 'dropout': 0.973845, 'filters': [[6, 6, 25]], 'pools': [2, 2]}, 2: {'activation': 'relu', 'dropout': 0.993991, 'filters': [[6, 6, 99]], 'pools': [2, 2]}, 3: {'activation': 'leaky_relu', 'dropout': 0.744906, 'filters': [[3, 3, 234]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'elu', 'dropout': 0.788669, 'size': 355}, 'init_stdev': 0.096934, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.000959}}
# {'conv': {1: {'activation': 'relu6', 'dropout': 0.972565, 'filters': [[3, 1, 39], [1, 3, 39]], 'pools': [2, 2]}, 2: {'activation': 'prelu', 'dropout': 0.913473, 'filters': [[7, 7, 108]], 'pools': [2, 2]}, 3: {'activation': 'prelu', 'dropout': 0.901391, 'filters': [[3, 3, 243]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'relu6', 'dropout': 0.629564, 'size': 662}, 'init_stdev': 0.063919, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.000990}}
def stage1():
  data = get_cifar10_data()

  curve_params = {
    'burn_in': 20,
    'min_input_size': 6,
    'value_limit': 0.5,
    'io_load_dir': '_models/cifar10/hyper/stage1-2.0',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.0',
  }
  curve_predictor = LinearCurvePredictor(**curve_params)

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 250,
      'eval_batch_size': 2500,
      'epochs': 15,
      'stop_condition': curve_predictor.stop_condition(),
      'result_metric': curve_predictor.result_metric(),
      'evaluate_test': True,
      'eval_flexible': False,
      'save_dir': '_models/cifar10/model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'save_accuracy_limit': 0.77,
    }

    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=data, runner=runner, **solver_params)
    return solver

  strategy_params = {
    'strategy': 'portfolio',
    'methods': ['ucb', 'pi', 'rand'],
    'probabilities': [0.3, 0.4, 0.3],
    'io_load_dir': '_models/cifar10/hyper/stage1-2.0',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.0',
  }

  tuner = HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
  tuner.tune()


if __name__ == "__main__":
  stage1()
