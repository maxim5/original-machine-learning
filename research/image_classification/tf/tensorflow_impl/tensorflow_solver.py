#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import os

import tensorflow as tf
from tensorflow.python.client import device_lib

from image_classification.tf.base_solver import BaseSolver
from image_classification.tf.util import *


class TensorflowSolver(BaseSolver):
  def __init__(self, data, runner, log_level=1, **params):
    self.load_dir = params.get('load_dir')
    self.saver = tf.train.Saver(defer_build=True)
    self.save_dir = params.get('save_dir')
    self.save_accuracy_limit = params.get('save_accuracy_limit', 0)

    self.session = None

    params['eval_flexible'] = params.get('eval_flexible', True) and _is_gpu_available
    super(TensorflowSolver, self).__init__(data, runner, log_level, **params)


  def prepare_data(self, data_set):
    data_set._epochs_completed = 0
    data_set._index_in_epoch = 0
    return data_set


  def prepare_runner(self, runner):
    if not self.load_dir:
      return runner
    hyper_params = self._load_dict(os.path.join(self.load_dir, 'hyper_params.xjson'))
    if hyper_params:
      self.info('Loaded hyper-params: %s' % dict_to_str(hyper_params))
      runner.hyper_params = hyper_params
    return runner


  def create_session(self):
    self.session = tf.Session()
    return self.session


  def init_session(self):
    results = self._load(self.load_dir, self.session, log_level=1)
    return results.get('validation_accuracy', 0)


  def on_best_accuracy(self, accuracy):
    super(TensorflowSolver, self).on_best_accuracy(accuracy)
    if accuracy > self.save_accuracy_limit:
      self._save(self.session, accuracy)


  def _evaluate_test(self):
    if not self.save_dir:
      return super(TensorflowSolver, self)._evaluate_test()

    # Load the best session if available before test evaluation
    current_results = self._load(self.save_dir, self.runner.session, log_level=0)
    eval = super(TensorflowSolver, self)._evaluate_test()
    if not current_results: return eval

    # Update the current results
    current_results['test_accuracy'] = eval.get('accuracy', 0)
    results_file = os.path.join(self.save_dir, 'results.xjson')
    with open(results_file, 'w') as file_:
      file_.write(dict_to_str(current_results))
      self.info('Results updated to %s' % results_file)
    return eval


  def _load(self, directory, session, log_level):
    if not directory:
      return {}

    directory = os.path.abspath(directory)
    session_file = os.path.join(directory, 'session.data')
    if os.path.exists(session_file):
      self.saver.build()
      self.saver.restore(session, session_file)
      self._log(log_level, 'Loaded session from %s' % session_file)

    results_file = os.path.join(directory, 'results.xjson')
    if os.path.exists(results_file):
      results = self._load_dict(results_file)
      self._log(log_level, 'Loaded results: %s from %s' % (dict_to_str(results), results_file))
      return results

    return {}


  def _save(self, session, accuracy):
    if not self.save_dir:
      return

    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)

    hyper_file = os.path.join(self.save_dir, 'hyper_params.xjson')
    results_file = os.path.join(self.save_dir, 'results.xjson')
    session_file = os.path.join(self.save_dir, 'session.data')

    self.saver.build()
    self.saver.save(session, session_file)
    self.debug('Session saved to %s' % session_file)

    # TODO: use self.runner.describe()

    with open(results_file, 'w') as file_:
      file_.write(dict_to_str({'validation_accuracy': accuracy, 'model_size': self.runner.model.params_num()}))
      self.debug('Results saved to %s' % results_file)

    with open(hyper_file, 'w') as file_:
      file_.write(dict_to_str(self.runner.hyper_params))
      self.debug('Hyper parameters saved to %s' % hyper_file)


  def _load_dict(self, from_file):
    if not os.path.exists(from_file):
      return {}
    try:
      with open(from_file, 'r') as file_:
        line = file_.readline()
        return str_to_dict(line)
    except:
      return {}


def tf_is_gpu():
  local_devices = device_lib.list_local_devices()
  return len([x for x in local_devices if x.device_type == 'GPU']) > 0


_is_gpu_available = tf_is_gpu()
