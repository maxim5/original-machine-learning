#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


class Kernel(object):
  def id(self, batch_x):
    pass

  def compute(self, batch_x, batch_y=None):
    pass


class RadialBasisFunction(Kernel):
  def __init__(self, gamma=0.5, **params):
    self.gamma = gamma
    self.params = params
    self.params.setdefault('metric', 'sqeuclidean')

  def id(self, batch_x):
    return np.ones(batch_x.shape[:1])

  def compute(self, batch_x, batch_y=None):
    if len(batch_x.shape) == 1:
      batch_x = batch_x.reshape((1, -1))
    if batch_y is not None and len(batch_y.shape) == 1:
      batch_y = batch_y.reshape((1, -1))

    if batch_y is None:
      dist = squareform(pdist(batch_x, **self.params))
    else:
      dist = cdist(batch_x, batch_y, **self.params)
    return np.exp(-self.gamma * dist)



########################################################################################################################

  # def compute_vector(self, batch, x):
  #   size = batch.shape[0]
  #   result = np.zeros(shape=(size, ))
  #   for i in xrange(size):
  #     result[i] = self.compute(batch[i], x)
  #   return result
  #
  # def compute_matrix(self, batch):
  #   size = batch.shape[0]
  #   result = np.zeros(shape=(size, size))
  #   for i in xrange(size):
  #     for j in xrange(i + 1):
  #       result[i, j] = result[j, i] = self.compute(batch[i], batch[j])
  #   return result

# class MyRadialBasisFunction(Kernel):
#   def __init__(self, gamma=0.5):
#     self.gamma = gamma
#
#   def compute(self, x, y):
#     diff = x - y
#     norm2 = np.sum(diff * diff)
#     return np.exp(-self.gamma * norm2)
#
#   def compute_vector(self, batch, x):
#     batch = batch.reshape((-1, x.shape[0]))
#     diff = batch - x
#     norm2 = np.sum(diff * diff, axis=1)
#     return np.exp(-self.gamma * norm2)
