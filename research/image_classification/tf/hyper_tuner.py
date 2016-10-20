#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import copy

from log import Logger
from util import *


def tf_reset_all():
  import tensorflow as tf
  tf.reset_default_graph()


class HyperTuner(Logger):
  def __init__(self, log_level=1):
    super(HyperTuner, self).__init__(log_level)
    self.best_accuracy = 0


  def _update_accuracy(self, trial, accuracy, tuned_params):
    marker = ' '
    if accuracy > self.best_accuracy:
      self.best_accuracy = accuracy
      marker = '!'
    self.info('%s [%d] accuracy=%.4f, params: %s' % (marker, trial, accuracy, dict_to_str(tuned_params)))


  def tune(self, solver_generator, fixed_params, tuned_params_generator):
    self.info('Start hyper-tuner')

    trial = 0
    while True:
      hyper_params = copy.deepcopy(fixed_params)
      tuned_params = tuned_params_generator()
      deep_update(hyper_params, tuned_params)

      solver = solver_generator(hyper_params)
      trial += 1
      tf_reset_all()
      accuracy = solver.train()
      self._update_accuracy(trial, accuracy, tuned_params)
