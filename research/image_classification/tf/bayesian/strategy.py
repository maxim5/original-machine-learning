#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np

from kernel import RadialBasisFunction
from maximizer import MonteCarloUtilityMaximizer
from strategy_io import StrategyIO
from utility import ProbabilityOfImprovement, ExpectedImprovement, UpperConfidenceBound, RandomPoint

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
  'rand': lambda points, values, kernel, mu_prior, params: RandomPoint(points, values, **slice_dict(params, 'rand_')),
}

maximizers = {
  'mc': lambda utility, sampler, params: MonteCarloUtilityMaximizer(utility, sampler, **slice_dict(params, 'mc_')),
}


class BaseStrategy(object):
  def __init__(self, sampler, **params):
    self._sampler = sampler
    self._params = params

    self._points = []
    self._values = []

    self._strategy_io = StrategyIO(self, **slice_dict(params, 'io_'))
    self._strategy_io.load()

  @property
  def iteration(self):
    return len(self._points)

  @property
  def points(self):
    return self._points

  @property
  def values(self):
    return self._values

  def next_proposal(self):
    raise NotImplementedError()

  def add_point(self, point, value):
    self._points.append(point)
    self._values.append(value)
    self._strategy_io.save()

  def import_from(self, data):
    self._points = data.get('points', [])
    self._values = data.get('values', [])

  def export_to(self):
    return {
      'points': [list(point) for point in self._points],
      'values': list(self._values),
    }


class BaseBayesianStrategy(BaseStrategy):
  def __init__(self, sampler, **params):
    super(BaseBayesianStrategy, self).__init__(sampler, **params)
    self._kernel = None
    self._maximizer = None

    self._mu_prior_gen = as_numeric_function(params.get('mu_prior_gen', 'mean'), presets=mu_priors)
    self._kernel_gen = as_function(params.get('kernel_gen', 'rbf'), presets=kernels)
    self._maximizer_gen = as_function(params.get('maximizer_gen', 'mc'), presets=maximizers)

    self._burn_in = params.get('burn_in', 1)

  def _instantiate(self, method_gen):
    mu_prior = self._mu_prior_gen(self._values)
    log('mu_prior=%.6f' % mu_prior)

    self._kernel = self._kernel_gen(self._params)
    self._method = method_gen(self._points, self._values, self._kernel, mu_prior, self._params)
    self._maximizer = self._maximizer_gen(self._method, self._sampler, self._params)

    return self._maximizer


class BayesianStrategy(BaseBayesianStrategy):
  def __init__(self, sampler, **params):
    super(BayesianStrategy, self).__init__(sampler, **params)
    self._method = None
    self._method_gen = as_function(params.get('utility_gen', 'ucb'), presets=utilities)

  def next_proposal(self):
    if self.iteration < self._burn_in:
      return self._sampler.sample(size=1)[0]

    self._maximizer = self._instantiate(self._method_gen)
    return self._maximizer.compute_max_point()


class BayesianPortfolioStrategy(BaseBayesianStrategy):
  def __init__(self, sampler, methods, **params):
    self._methods = [as_function(gen, presets=utilities) for gen in methods]
    self._probabilities = params.get('probabilities')
    self._scores = np.zeros(shape=len(methods))
    self._index = None
    self._alpha = params.get('alpha', 0.9)

    super(BayesianPortfolioStrategy, self).__init__(sampler, **params)

  def next_proposal(self):
    if self.iteration < self._burn_in:
      return self._sampler.sample(size=1)[0]

    if self._probabilities is None:
      self._probabilities = softmax(self._scores)
    self._index = np.random.choice(range(len(self._scores)), p=self._probabilities)
    method_gen = self._methods[self._index]
    self._maximizer = self._instantiate(method_gen)
    return self._maximizer.compute_max_point()

  def add_point(self, point, value):
    if self.iteration >= self._burn_in and self._probabilities is None:
      score = value - np.mean(self._values)
      self._scores[self._index] = self._alpha * score + (1 - self._alpha) * self._scores[self._index]

    super(BayesianPortfolioStrategy, self).add_point(point, value)

  def import_from(self, data):
    self._points = data.get('points', [])
    self._values = data.get('values', [])
    self._scores = data.get('scores', np.zeros(shape=len(self._methods)))

  def export_to(self):
    return {
      'points': [list(point) for point in self._points],
      'values': list(self._values),
      'scores': list(self._scores),
    }


def softmax(scores):
  scores -= np.mean(scores)
  exp = np.exp(scores)
  return exp / np.sum(exp)
