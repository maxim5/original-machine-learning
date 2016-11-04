#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
from scipy import stats


class Utility(object):
  def __init__(self, points, values):
    super(Utility, self).__init__()
    self.points = np.array(points)
    self.values = np.array(values)
    assert self.points.shape[0] == self.values.shape[0]

  def compute_values(self, batch):
    pass


class BaseGaussianUtility(Utility):
  def __init__(self, points, values, kernel, **params):
    super(BaseGaussianUtility, self).__init__(points, values)
    self.kernel = kernel

    kernel_matrix = self.kernel.compute(self.points)
    self.k_inv = np.linalg.pinv(kernel_matrix)
    self.k_inv_f = np.dot(self.k_inv, self.values)

  def _mean_and_std(self, batch):
    batch = np.array(batch)
    k_star = np.swapaxes(self.kernel.compute(self.points, batch), 0, 1)
    k_star_star = self.kernel.id(batch)

    mu_star = np.dot(k_star, self.k_inv_f)

    t_star = np.dot(self.k_inv, k_star.T)
    t_star = np.einsum('ij,ji->i', k_star, t_star)
    sigma_star = k_star_star - t_star

    return mu_star, sigma_star


class ProbabilityOfImprovement(BaseGaussianUtility):
  def __init__(self, points, values, kernel, **params):
    super(ProbabilityOfImprovement, self).__init__(points, values, kernel, **params)
    self.epsilon = params.get('epsilon', 1e-8)
    self.max_value = np.max(self.values)

  def compute_values(self, batch):
    mu_batch, sigma_batch = self._mean_and_std(batch)
    z_batch = (mu_batch - self.max_value - self.epsilon) / sigma_batch
    cdf = stats.norm.cdf(z_batch)
    cdf[np.abs(mu_batch - self.max_value) < self.epsilon] = 0.0
    return cdf
