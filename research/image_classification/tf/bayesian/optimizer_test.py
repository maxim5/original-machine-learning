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
    self.run_opt(a=-10, b=10, start=5, f=lambda x: np.abs(np.sin(x)/x), global_max=1, steps=15, plot=False)
    self.run_opt(a=-10, b=10, start=5, f=lambda x: x*x, global_max=100, steps=15, plot=False)
    self.run_opt(a=-10, b=10, start=5, f=lambda x: np.sin(np.log(np.abs(x))), global_max=1, steps=15, plot=False)
    self.run_opt(a=-8, b=8, start=3, f=lambda x: x/(np.sin(x)+2), global_max=4.8, steps=15, plot=False)

  def test_1d_complex(self):
    self.run_opt(a=-10, b=10, start=3, f=lambda x: x * np.sin(x), global_max=7.9, steps=50, plot=False)
    self.run_opt(a=-12, b=16, start=3, f=lambda x: x*np.sin(x+1)/2, global_max=6.55, steps=50, plot=False)
    self.run_opt(a=0, b=10, start=3, f=lambda x: np.exp(np.sin(x*5)*np.sqrt(x)), global_max=20.3, steps=50, plot=False)

  def test_2d_simple(self):
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: x[0] + x[1], global_max=20, steps=15, plot=False)
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0]) + np.cos(x[1]), global_max=2, steps=15, plot=False)

  def test_2d_complex(self):
    self.run_opt(a=(0, 0), b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0]) * np.cos(x[1]), global_max=1, steps=50, plot=False)


  def run_opt(self, a, b, f, global_max, start=None, steps=10, plot=False):
    sampler = DefaultSampler()
    sampler.add(lambda: np.random.uniform(a, b))

    self.optimizer = BayesianOptimizer(sampler, batch_size=10000)

    def add(x):
      self.optimizer.add_point(np.asarray(x), f(x))

    if start is not None:
      add(start)
    for i in xrange(steps):
      x = self.optimizer.next_proposal()
      log('selected_point=%s -> true=%.6f' % (x, f(x)))
      add(x)

    i = np.argmax(self.optimizer.values)
    log('Best found: %s -> %.6f' % (self.optimizer.points[i], self.optimizer.values[i]))

    if plot:
      if type(a) in [tuple, list]:
        self._plot_2d(a, b, f)
      else:
        self._plot_1d(a, b, f)

    delta = abs(global_max) / 10.0
    self.assertAlmostEqual(self.optimizer.values[i], global_max, delta=delta)


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


  def _debug(self, val):
    log(self.optimizer.utility.compute_values(np.asarray([[val]])))
    log(self.optimizer.utility.compute_values(self.optimizer.utility.points))


if __name__ == "__main__":
  unittest.main()
