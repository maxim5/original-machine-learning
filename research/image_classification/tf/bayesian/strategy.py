#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np

from kernel import RadialBasisFunction
from maximizer import MonteCarloUtilityMaximizer
from strategy_io import StrategyIO
from utility import ProbabilityOfImprovement, ExpectedImprovement, UpperConfidenceBound

from image_classification.tf.log import log
from image_classification.tf.util import as_function, as_numeric_function, slice_dict


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

class BayesianStrategy(object):
  def __init__(self, sampler, **params):
    self.sampler = sampler
    self.kernel = None
    self.utility = None
    self.maximizer = None

    self.params = params
    self.mu_prior_gen = as_numeric_function(params.get('mu_prior_gen', 'mean'), presets=mu_priors)
    self.kernel_gen = as_function(params.get('kernel_gen', 'rbf'), presets=kernels)
    self.utility_gen = as_function(params.get('utility_gen', 'ucb'), presets=utilities)
    self.maximizer_gen = as_function(params.get('maximizer_gen', 'mc'), presets=maximizers)

    self.strategy_io = StrategyIO(self, **slice_dict(params, 'io_'))
    self.points, self.values = self.strategy_io.load()

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
    self.strategy_io.save()
