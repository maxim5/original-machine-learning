#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numbers
import numpy as np

from kernel import RadialBasisFunction
from maximizer import MonteCarloUtilityMaximizer
from utility import ProbabilityOfImprovement, ExpectedImprovement, UpperConfidenceBound

from image_classification.tf.log import log


mu_priors = {
  'mean': lambda values: np.mean(values, axis=0),
  'max': lambda values: np.max(values, axis=0),
}

kernels = {
  'rbf': lambda params: RadialBasisFunction(**slice_dict(params, 'rbf_')),
}

utilities = {
  'pi': lambda points, values, kernel, mu_prior, params: ProbabilityOfImprovement(points, values, kernel, mu_prior,
                                                                                  **slice_dict(params, 'pi_')),
  'ei': lambda points, values, kernel, mu_prior, params: ExpectedImprovement(points, values, kernel, mu_prior,
                                                                             **slice_dict(params, 'ei_')),
  'ucb': lambda points, values, kernel, mu_prior, params: UpperConfidenceBound(points, values, kernel, mu_prior,
                                                                               **slice_dict(params, 'ucb_')),
}

maximizers = {
  'mc': lambda utility, sampler, params: MonteCarloUtilityMaximizer(utility, sampler, **slice_dict(params, 'mc_')),
}

class BayesianOptimizer(object):
  def __init__(self, sampler, **params):
    self.sampler = sampler
    self.points = []
    self.values = []
    self.kernel = None
    self.utility = None
    self.maximizer = None

    self.params = params
    self.mu_prior_gen = as_numeric_function(params.get('mu_prior_gen', 'mean'), presets=mu_priors)
    self.kernel_gen = as_function(params.get('kernel_gen', 'rbf'), presets=kernels)
    self.utility_gen = as_function(params.get('utility_gen', 'ucb'), presets=utilities)
    self.maximizer_gen = as_function(params.get('maximizer_gen', 'mc'), presets=maximizers)

  def next_proposal(self):
    if not self.points:
      return self.sampler.sample(size=1)[0]

    mu_prior = self.mu_prior_gen(self.values)
    log('mu_prior=%.6f' % mu_prior)

    self.kernel = self.kernel_gen(self.params)
    self.utility = self.utility_gen(self.points, self.values, self.kernel, mu_prior, self.params)
    self.maximizer = self.maximizer_gen(self.utility, self.sampler, self.params)

    return self.maximizer.compute_max_point()

  def add_point(self, point, value):
    self.points.append(point)
    self.values.append(value)

def as_function(val, presets, default=None):
  if callable(val):
    return val

  preset = presets.get(val, default)
  if preset is not None:
    return preset

  raise ValueError('Value is not recognized: ', val)

def as_numeric_function(val, presets, default=None):
  if isinstance(val, numbers.Number):
    def const(*_):
      return val
    return const

  return as_function(val, presets, default)

def slice_dict(d, key_prefix):
  return {key[len(key_prefix):]: value for key, value in d.iteritems() if key.startswith(key_prefix)}
