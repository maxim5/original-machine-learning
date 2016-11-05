#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
from image_classification.tf.log import log


class BaseUtilityMaximizer(object):
  def __init__(self, utility):
    super(BaseUtilityMaximizer, self).__init__()
    self.utility = utility

  def compute_max_point(self):
    pass


class MonteCarloUtilityMaximizer(BaseUtilityMaximizer):
  def __init__(self, utility, sampler, **params):
    super(MonteCarloUtilityMaximizer, self).__init__(utility)
    self.sampler = sampler
    self.batch_size = params.get('batch_size', 100000)

  def compute_max_point(self):
    batch = self.sampler.sample(size=self.batch_size)
    values = self.utility.compute_values(batch)
    i = np.argmax(values)
    log('Max prediction_value: %.6f' % values[i])
    return batch[i]