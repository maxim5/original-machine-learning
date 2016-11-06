#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np

from kernel import RadialBasisFunction
from maximizer import MonteCarloUtilityMaximizer
from utility import UpperConfidenceBound

from image_classification.tf.log import log


class BayesianOptimizer(object):
  def __init__(self, sampler, **params):
    self.sampler = sampler
    self.points = []
    self.values = []
    self.kernel = None
    self.utility = None
    self.maximizer = None
    self.params = params

  def next_proposal(self):
    if not self.points:
      return self.sampler.sample(size=1)[0]

    mu_prior = 0
    if True:
      mu_prior = np.mean(self.values, axis=0)
    log('mu_prior=%.6f' % mu_prior)

    self.kernel = RadialBasisFunction()
    self.utility = UpperConfidenceBound(self.points, self.values, self.kernel, mu_prior=mu_prior)
    self.maximizer = MonteCarloUtilityMaximizer(self.utility, self.sampler, **self.params)
    return self.maximizer.compute_max_point()

  def add_point(self, point, value):
    self.points.append(point)
    self.values.append(value)
