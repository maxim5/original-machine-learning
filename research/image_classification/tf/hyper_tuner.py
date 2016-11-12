#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from log import Logger
from util import *
from image_classification.tf.spec.parsed_spec import ParsedSpec

from bayesian.sampler import DefaultSampler
from bayesian.optimizer import BayesianOptimizer


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


  def tune(self, solver_generator, hyper_params_spec):
    self.info('Start hyper-tuner')

    parsed = ParsedSpec(hyper_params_spec)
    self.info('Spec size=%d' % parsed.size())

    sampler = DefaultSampler()
    sampler.add_uniform(parsed.size())
    optimizer = BayesianOptimizer(sampler, mc_batch_size=100000)

    trial = 0
    while True:
      point = optimizer.next_proposal()
      hyper_params = parsed.instantiate(point)
      solver = solver_generator(hyper_params)
      trial += 1
      tf_reset_all()
      accuracy = solver.train()
      self._update_accuracy(trial, accuracy, hyper_params)
      optimizer.add_point(point, accuracy)
