#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from kernel import RadialBasisFunction
from maximizer import MonteCarloUtilityMaximizer
from utility import ProbabilityOfImprovement


class BayesianOptimizer(object):
  def __init__(self, sampler):
    self.sampler = sampler
    self.points = []
    self.values = []
    self.kernel = None
    self.utility = None
    self.maximizer = None

  def next_proposal(self):
    if not self.points:
      return self.sampler.sample(size=1)[0]

    self.kernel = RadialBasisFunction()
    self.utility = ProbabilityOfImprovement(self.points, self.values, self.kernel)
    self.maximizer = MonteCarloUtilityMaximizer(self.utility, self.sampler)
    return self.maximizer.compute_max_point()

  def add_point(self, point, value):
    self.points.append(point)
    self.values.append(value)
