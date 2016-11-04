#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np


class UtilityMaximizer(object):
  def __init__(self, utility):
    super(UtilityMaximizer, self).__init__()
    self.utility = utility

  def compute_max_point(self):
    pass


class MonteCarloUtilityMaximizer(UtilityMaximizer):
  def __init__(self, utility, sampler, **params):
    super(MonteCarloUtilityMaximizer, self).__init__(utility)
    self.sampler = sampler
    self.batch_size = params.get('batch_size', 100000)

  def compute_max_point(self):
    batch = self.sampler.sample(size=self.batch_size)
    values = self.utility.compute_values(batch)
    i = np.argmax(values)
    return batch[i]
