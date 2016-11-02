#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
from scipy import stats

from log import Logger


class Utility(object):
  def __init__(self, points, values):
    super(Utility, self).__init__()
    self.points = np.array(points)
    self.values = np.array(values)
    assert self.points.shape[0] == self.values.shape[0]


  def compute_point_value(self, x):
    pass


def rbf_kernel(x, y, gamma=0.5):
  norm2 = np.sum(x*x, axis=0) + np.sum(y*y, axis=0) - 2*np.dot(x, y.T)
  return np.exp(-gamma * norm2)


class BaseGaussianUtility(Utility):
  def __init__(self, points, values, kernel_function, **params):
    super(BaseGaussianUtility, self).__init__(points, values)
    self.kernel_function = kernel_function

    kernel_matrix = self._compute_matrix(points)
    self.k_inv = np.linalg.pinv(kernel_matrix)
    self.k_inv_f = np.dot(self.k_inv, self.values)


  def _mean_and_std(self, x):
    x = np.array(x)
    k_star = self._compute_matrix(self.points, x.reshape((1, -1)))
    k_star_star = self.kernel_function(x, x)
    mu_star = np.dot(k_star.T, self.k_inv_f)
    sigma_star = k_star_star - np.dot(k_star.T, np.dot(self.k_inv, k_star))
    return mu_star, sigma_star


  def _compute_matrix(self, x, y=None):
    dim_x = x.shape[0]
    dim_y = y.shape[0] if y is not None else 0
    if dim_y == 0:
      result = np.zeros(shape=(dim_x, dim_x))
      for i in xrange(dim_x):
        for j in xrange(i + 1):
          result[i, j] = result[j, i] = self.kernel_function(x[i], x[j])
      return result
    elif dim_y == 1:
      result = np.zeros(shape=(dim_x, ))
      for i in xrange(dim_x):
        result[i] = self.kernel_function(x[i], y[0])
      return result
    else:
      result = np.zeros(shape=(dim_x, dim_y))
      for i in xrange(dim_x):
        for j in xrange(dim_y):
          result[i, j] = self.kernel_function(x[i], y[j])
      return result


class ProbabilityOfImprovement(BaseGaussianUtility):
  def __init__(self, points, values, kernel_function, **params):
    super(ProbabilityOfImprovement, self).__init__(points, values, kernel_function, **params)
    self.epsilon = params.get('epsilon', 1e-8)
    self.max_value = np.max(self.values)


  def compute_point_value(self, x):
    mu, sigma = self._mean_and_std(x)
    if abs(mu - self.max_value) < self.epsilon:
      return 0.0
    z = (mu - self.max_value - self.epsilon) / sigma
    return stats.norm.cdf(z)


class UtilityMaximizer(object):
  def __init__(self, utility):
    super(UtilityMaximizer, self).__init__()
    self.utility = utility

  def compute_max_point(self):
    pass


class Sampler(object):
  def sample(self):
    pass


class MonteCarloUtilityMaximizer(UtilityMaximizer):
  def __init__(self, utility, sampler, **params):
    super(MonteCarloUtilityMaximizer, self).__init__(utility)
    self.sampler = sampler

    self.sample_size = params.get('sample_size', 1000)

  def compute_max_point(self):
    max_value = 0
    max_point = None
    for i in xrange(self.sample_size):
      point = self.sampler.sample()
      value = self.utility.compute_point_value(point)
      if value > max_value:
        max_value = value
        max_point = point
    return max_point, max_value


class BayesianOptimizer(Logger):
  def __init__(self, log_level=1):
    super(BayesianOptimizer, self).__init__(log_level)



def test1():
  x1 = (0.0, 1.0, 0.5, 3.0)
  x2 = (0.2, 1.2, 0.5, 4.0)
  x3 = (0.1, 1.5, 1.0, 2.5)
  x4 = (0.1, 1.0, 0.7, 2.7)
  x = np.array([x1, x2, x3, x4])

  f1 = 0.6
  f2 = 0.4
  f3 = 0.5
  f4 = 0.4
  f = np.array([f1, f2, f3, f4])

  util = ProbabilityOfImprovement(points=x, values=f, kernel_function=rbf_kernel)

  print 'around x1'
  print util.compute_point_value(x=(0.0, 0.0, 0.0, 0.0))
  print util.compute_point_value(x=(0.0, 0.0, 0.0, 3.0))
  print util.compute_point_value(x=(0.0, 0.0, 0.5, 3.0))
  print util.compute_point_value(x=(0.0, 0.5, 0.5, 3.0))
  print util.compute_point_value(x=(0.0, 0.8, 0.5, 3.0))
  print util.compute_point_value(x=(0.0, 0.9, 0.5, 3.0))
  print util.compute_point_value(x=(0.0, 0.99, 0.5, 3.0))
  print util.compute_point_value(x=(0.0, 1.0, 0.5, 3.0))

  print 'far away'
  print util.compute_point_value(x=(0.0, 2.0, 0.0, 0.0))
  print util.compute_point_value(x=(0.0, -2.0, 0.0, 0.0))
  print util.compute_point_value(x=(0.0, -5.0, 0.0, 0.0))
  print util.compute_point_value(x=(0.0, -5.0, -1.0, 0.0))
  print util.compute_point_value(x=(0.0, 2.0, -1.0, 3.0))
  print util.compute_point_value(x=(0.0, 2.0, 0.3, 3.5))


def test2():
  x1 = (0.0, 1.0, 0.5, 3.0)
  x2 = (0.2, 1.2, 0.5, 4.0)
  x3 = (0.1, 1.5, 1.0, 2.5)
  x4 = (0.1, 1.0, 0.7, 2.7)
  x = np.array([x1, x2, x3, x4])

  f1 = 0.6
  f2 = 0.4
  f3 = 0.5
  f4 = 0.4
  f = np.array([f1, f2, f3, f4])

  util = ProbabilityOfImprovement(points=x, values=f, kernel_function=rbf_kernel)

  class MySampler(Sampler):
    def sample(self):
      return [np.random.uniform(0, 3), np.random.uniform(1, 2), np.random.uniform(0, 2), np.random.uniform(0, 5)]

  sampler = MySampler()
  maximizer = MonteCarloUtilityMaximizer(util, sampler)

  for i in xrange(10):
    print maximizer.compute_max_point()


def rand_x(size):
  return np.random.standard_normal(size)

def rand_pi(size=4, num=10):
  x = rand_x((num, size))
  f = rand_x(num) * 2
  return ProbabilityOfImprovement(points=x, values=f, kernel_function=rbf_kernel)

def benchmark1():
  import timeit
  print timeit.timeit('x = rand_x(4);   rbf_kernel(x, x)', 'from __main__ import rbf_kernel, rand_x', number=10000)
  print timeit.timeit('x = rand_x(8);   rbf_kernel(x, x)', 'from __main__ import rbf_kernel, rand_x', number=10000)
  print timeit.timeit('x = rand_x(50);  rbf_kernel(x, x)', 'from __main__ import rbf_kernel, rand_x', number=10000)
  print timeit.timeit('x = rand_x(100); rbf_kernel(x, x)', 'from __main__ import rbf_kernel, rand_x', number=10000)
  print timeit.timeit('x = rand_x(200); rbf_kernel(x, x)', 'from __main__ import rbf_kernel, rand_x', number=10000)

def benchmark2():
  import timeit
  print timeit.timeit('pi = rand_pi(4,  10); x = rand_x(4);  pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)
  print timeit.timeit('pi = rand_pi(8,  10); x = rand_x(8);  pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)
  print timeit.timeit('pi = rand_pi(20, 10); x = rand_x(20); pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)
  print timeit.timeit('pi = rand_pi(50, 10); x = rand_x(50); pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)
  print timeit.timeit('pi = rand_pi(50, 20); x = rand_x(50); pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)
  print timeit.timeit('pi = rand_pi(50, 80); x = rand_x(50); pi.compute_point_value(x)', 'from __main__ import rbf_kernel, rand_x, rand_pi', number=1000)

if __name__ == "__main__":
  benchmark2()
