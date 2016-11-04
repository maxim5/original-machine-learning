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

  def next_proposal(self):
    if not self.points:
      return self.sampler.sample(size=1)[0]

    kernel = RadialBasisFunction()
    utility = ProbabilityOfImprovement(self.points, self.values, kernel)
    maximizer = MonteCarloUtilityMaximizer(utility, self.sampler)
    return maximizer.compute_max_point()

  def add_point(self, point, value):
    self.points.append(point)
    self.values.append(value)
