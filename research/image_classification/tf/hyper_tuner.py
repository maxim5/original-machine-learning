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
  def __init__(self, hyper_params_spec, solver_generator, **strategy_params):
    super(HyperTuner, self).__init__(1)
    self.best_accuracy = 0
    self.solver_generator = solver_generator

    self.parsed = ParsedSpec(hyper_params_spec)
    self.info('Spec size=%d' % self.parsed.size())

    sampler = DefaultSampler()
    sampler.add_uniform(self.parsed.size())
    self.strategy = BayesianOptimizer(sampler, **strategy_params)


  def tune(self, ):
    self.info('Start hyper-tuner')

    trial = 0
    while True:
      point = self.strategy.next_proposal()
      hyper_params = self.parsed.instantiate(point)
      solver = self.solver_generator(hyper_params)
      trial += 1
      tf_reset_all()
      accuracy = solver.train()
      self._update_accuracy(trial, accuracy, hyper_params)
      self.strategy.add_point(point, accuracy)


  def _update_accuracy(self, trial, accuracy, tuned_params):
    marker = ' '
    if accuracy > self.best_accuracy:
      self.best_accuracy = accuracy
      marker = '!'
    self.info('%s [%d] accuracy=%.4f, params: %s' % (marker, trial, accuracy, dict_to_str(tuned_params)))
