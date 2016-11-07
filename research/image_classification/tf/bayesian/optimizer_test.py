#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from optimizer import BayesianOptimizer
from sampler import DefaultSampler

from image_classification.tf.log import log

class BayesianOptimizerTest(unittest.TestCase):
  def test_1d_simple(self):
    self.run_opt(a=-10, b=10, start=5, f=lambda x: np.abs(np.sin(x)/x), global_max=1, steps=10, plot=False)
    self.run_opt(a=-10, b=10, start=5, f=lambda x: x*x, global_max=100, steps=10, plot=False)
    self.run_opt(a=-10, b=10, start=5, f=lambda x: np.sin(np.log(np.abs(x))), global_max=1, steps=10, plot=False)
    self.run_opt(a=-8, b=8, start=3, f=lambda x: x/(np.sin(x)+2), global_max=4.8, steps=10, plot=False)

  def test_1d_complex(self):
    self.run_opt(a=-10, b=10, start=3, f=lambda x: x*np.sin(x), global_max=7.9, steps=30, plot=False)
    self.run_opt(a=-12, b=16, start=3, f=lambda x: x*np.sin(x+1)/2, global_max=6.55, steps=30, plot=False)
    self.run_opt(a=0, b=10, start=3, f=lambda x: np.exp(np.sin(x*5)*np.sqrt(x)), global_max=20.3, steps=30, plot=False)

  def test_2d_simple(self):
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: x[0]+x[1], global_max=20, steps=10, plot=False)
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0])+np.cos(x[1]), global_max=2, steps=10, plot=False)

  def test_2d_multiple_local_maxima(self):
    self.run_opt(a=(0, 0), b=(9, 9), start=None, f=(lambda x: (x[0]+x[1])/(np.exp(-np.sin(x[0])))), global_max=46, steps=20, plot=False)
    self.run_opt(a=(0, 0), b=(9, 9), start=None, f=(lambda x: (x[0]+x[1])/((x[0]-1)**2-np.sin(x[1])+2)), global_max=8.95, steps=20, plot=False)
    self.run_opt(a=(-10, -10), b=(15, 15), start=None, f=lambda x: np.sum(x*np.sin(x+1)/2, axis=0), global_max=13.175, steps=40, plot=False)

  def test_2d_complex(self):
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0])*np.cos(x[1]), global_max=1, steps=30, plot=False)
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0])/(np.cos(x[1])+2), global_max=1, steps=40, plot=False)

  def run_opt(self, a, b, f, global_max, start=None, steps=10, plot=False):
    if plot:
      self._run(a, b, f, start, steps, batch_size=100000, stop_condition=None)
      self._plot(a, b, f)
      return

    size_list = [1000, 10000, 50000, 100000]
    for batch_size in size_list:
      try:
        delta = abs(global_max) / 10.0
        max_value = self._run(a, b, f, start, steps, batch_size,
                              stop_condition=lambda x: abs(f(x) - global_max) <= delta)
        self.assertAlmostEqual(max_value, global_max, delta=delta)
        return
      except AssertionError as e:
        log('Attempt for %d failed: %s' % (batch_size, str(e)))
        if batch_size == size_list[-1]:
          raise

  def _run(self, a, b, f, start, steps, batch_size, stop_condition):
    sampler = DefaultSampler()
    sampler.add(lambda: np.random.uniform(a, b))
    self.optimizer = BayesianOptimizer(sampler, batch_size=batch_size)

    if start is not None:
      self.optimizer.add_point(np.asarray(start), f(start))

    for i in xrange(steps):
      x = self.optimizer.next_proposal()
      log('selected_point=%s -> true=%.6f' % (x, f(x)))
      self.optimizer.add_point(x, f(x))
      if stop_condition is not None and stop_condition(x):
        break

    i = np.argmax(self.optimizer.values)
    log('Best found: %s -> %.6f' % (self.optimizer.points[i], self.optimizer.values[i]))
    return self.optimizer.values[i]

  def _plot(self, a, b, f):
    if type(a) in [tuple, list]:
      assert len(a) == 2
      self._plot_2d(a, b, f)
    else:
      self._plot_1d(a, b, f)

  def _plot_1d(self, a, b, f, grid_size=1000):
    grid = np.linspace(a, b, num=grid_size).reshape((-1, 1))
    mu, sigma = self.optimizer.utility.mean_and_std(grid)

    plt.plot(grid, f(grid), color='black', linewidth=1.5, label='f')
    plt.plot(grid, mu, color='red', label='mu')
    plt.plot(grid, mu + sigma, color='blue', linewidth=0.4, label='mu+sigma')
    plt.plot(grid, mu - sigma, color='blue', linewidth=0.4)
    plt.plot(self.optimizer.points, f(np.asarray(self.optimizer.points)), 'o', color='red')
    plt.xlim([a - 0.5, b + 0.5])
    # plt.legend()
    plt.show()

  def _plot_2d(self, a, b, f, grid_size=200):
    grid_x = np.linspace(a[0], b[0], num=grid_size).reshape((-1, 1))
    grid_y = np.linspace(a[1], b[1], num=grid_size).reshape((-1, 1))
    x, y = np.meshgrid(grid_x, grid_y)

    merged = np.stack([x.flatten(), y.flatten()])
    z = f(merged).reshape(x.shape)

    swap = np.swapaxes(merged, 0, 1)
    mu, sigma = self.optimizer.utility.mean_and_std(swap)
    mu = mu.reshape(x.shape)
    sigma = sigma.reshape(x.shape)

    points = np.asarray(self.optimizer.points)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = f(np.swapaxes(points, 0, 1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, color='black', label='f', alpha=0.7,
                    linewidth=0, antialiased=False)
    ax.plot_surface(x, y, mu, color='red', label='mu', alpha=0.5)
    ax.plot_surface(x, y, mu + sigma, color='blue', label='mu+sigma', alpha=0.3)
    ax.plot_surface(x, y, mu - sigma, color='blue', alpha=0.3)
    ax.scatter(xs, ys, zs, color='red', marker='o', s=100)
    # plt.legend()
    plt.show()

  def _eval_max(self, a, b, f):
    sampler = DefaultSampler()
    sampler.add(lambda: np.random.uniform(a, b))
    batch = sampler.sample(1000000)
    batch = np.swapaxes(batch, 1, 0)
    return np.max(f(batch))

  def _debug(self, val):
    log(self.optimizer.utility.compute_values(np.asarray([[val]])))
    log(self.optimizer.utility.compute_values(self.optimizer.utility.points))


if __name__ == "__main__":
  unittest.main()
