#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


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


  def tune(self, solver_generator, tuned_params_generator):
    self.info('Start hyper-tuner')

    trial = 0
    while True:
      hyper_params = tuned_params_generator()
      solver = solver_generator(hyper_params)
      trial += 1
      tf_reset_all()
      accuracy = solver.train()
      self._update_accuracy(trial, accuracy, hyper_params)


  def tune2(self, solver_generator, tuned_params_generator):
    self.info('Start HYPE')

    bag = []
    _uniform = np.random.uniform
    _choice = np.random.choice

    def uniform_patch(low=0.0, high=1.0):
      result = _uniform(low, high)
      norm = (result - low) / (high - low)
      bag.append(norm)
      return result

    def choice_patch(a):
      result = _choice(a)
      idx = a.index(result)
      norm = float(idx) / len(a)
      bag.append(norm)
      return result

    np.random.uniform = uniform_patch
    np.random.choice = choice_patch

    from gaussian_optimization import Sampler, BayesianOptimizer
    class MySampler(Sampler):
      def __init__(self):
        self.map = {}

      def sample_one(self):
        global bag
        bag = []
        hyper_params = tuned_params_generator()
        self.map[sum(bag)] = hyper_params
        return bag

      def sample_batch(self, size):
        self.map = {}
        return super(MySampler, self).sample_batch(size)

      def get(self, bag):
        return self.map[sum(bag)]

    sampler = MySampler()
    optimizer = BayesianOptimizer(sampler)

    trial = 0
    while True:
      point = optimizer.next_proposal()
      hyper_params = sampler.get(point)
      solver = solver_generator(hyper_params)
      trial += 1
      tf_reset_all()
      accuracy = solver.train()
      self._update_accuracy(trial, accuracy, hyper_params)
      optimizer.add_point(point, accuracy)
