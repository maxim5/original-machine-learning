#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import datetime
import sys

from tflearn.datasets import cifar10

from image_classification.tf.conv_model import ConvModel
from image_classification.tf.hyper_tuner import HyperTuner
from image_classification.tf.tensorflow_impl import TensorflowRunner, TensorflowSolver
from image_classification.tf.data_set import Data, DataSet
from image_classification.tf.util import *
from image_classification.tf.curve_predictor import LinearCurvePredictor
from image_classification.tf.interaction import list_models

from cifar_spec import hyper_params_spec_2_0, hyper_params_spec_2_5


def get_cifar10_data(validation_size=5000, one_hot=True):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data('../../../dat/cifar-10-tf', one_hot=one_hot)

  x_val = x_train[:validation_size]
  y_val = y_train[:validation_size]
  x_train = x_train[validation_size:]
  y_train = y_train[validation_size:]

  return Data(train=DataSet(x_train, y_train),
              validation=DataSet(x_val, y_val),
              test=DataSet(x_test, y_test))


def stage1():
  data = get_cifar10_data()

  curve_params = {
    'burn_in': 20,
    'min_input_size': 5,
    'value_limit': 0.5,
    'io_load_dir': '_models/cifar10/hyper/stage1-2.5',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.5',
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
    'probabilities': [0.6, 0.1, 0.3],
    'io_load_dir': '_models/cifar10/hyper/stage1-2.5',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.5',
  }

  tuner = HyperTuner(hyper_params_spec_2_5, solver_generator, **strategy_params)
  tuner.tune()


def stage2():
  data = get_cifar10_data()
  hyper_params = {}

  solver_params = {
    'batch_size': 250,
    'eval_batch_size': 2500,
    'epochs': 16,
    'evaluate_test': True,
    'eval_flexible': True,
    'save_dir': '_models/cifar10/model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
    'save_accuracy_limit': 0.76,
  }

  model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  solver = TensorflowSolver(data=data, runner=runner, **solver_params)
  solver.train()


def list_all(path='_models/cifar10/model-zoo'):
  list_models(path)


if __name__ == "__main__":
  run_config = {
    'stage1': stage1,
    'stage2': stage2,
    'list': list_all,
  }

  arguments = sys.argv
  method = run_config.get(arguments[1])
  args = arguments[2:]
  method(*args)
