#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf
from tensorflow.python.client import device_lib

from image_classification.tf.base_solver import BaseSolver
from tensorflow_model_io import TensorflowModelIO


class TensorflowSolver(BaseSolver):
  def __init__(self, data, runner, augmentation=None, model_io=None, log_level=1, **params):
    self.session = None
    self.model_io = model_io if model_io is not None else TensorflowModelIO(log_level, **params)
    self.save_accuracy_limit = params.get('save_accuracy_limit', 0)

    params['eval_flexible'] = params.get('eval_flexible', True) and _is_gpu_available
    super(TensorflowSolver, self).__init__(data, runner, augmentation, log_level, **params)


  def create_session(self):
    self.session = tf.Session()
    return self.session


  def init_session(self):
    results = self._load(directory=self.model_io.load_dir, log_level=1)
    return results.get('validation_accuracy', 0)


  def on_best_accuracy(self, accuracy, eval_result):
    super(TensorflowSolver, self).on_best_accuracy(accuracy, eval_result)
    if accuracy >= self.save_accuracy_limit:
      runner_describe = self.runner.describe()
      self.model_io.save_results({'validation_accuracy': accuracy, 'model_size': runner_describe.get('model_size', 0)})
      self.model_io.save_hyper_params(runner_describe.get('hyper_params', {}))
      self.model_io.save_session(self.session)
      self.model_io.save_data(eval_result.get('data'))


  def _evaluate_test(self):
    # Load the best session if available before test evaluation
    current_results = self._load(directory=self.model_io.save_dir, log_level=0)
    eval_ = super(TensorflowSolver, self)._evaluate_test()
    if not current_results:
      return eval_

    # Update the current results
    current_results['test_accuracy'] = eval_.get('accuracy', 0)
    self.model_io.save_results(current_results)
    return eval_


  def _load(self, directory, log_level):
    self.model_io.load_session(self.session, directory, log_level)
    results = self.model_io.load_results(directory, log_level)
    return results or {}


def tf_is_gpu():
  local_devices = device_lib.list_local_devices()
  return len([x for x in local_devices if x.device_type == 'GPU']) > 0


_is_gpu_available = tf_is_gpu()
