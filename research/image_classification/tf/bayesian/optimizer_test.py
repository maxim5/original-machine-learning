#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import math
import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from optimizer import BayesianOptimizer
from sampler import DefaultSampler

from image_classification.tf.log import log

class BayesianOptimizerTest(unittest.TestCase):
  # 1-D

  def test_1d_simple(self):
    self.run_opt(f=lambda x: np.abs(np.sin(x) / x), a=-10, b=10, start=5, global_max=1, steps=10, plot=False)
    self.run_opt(f=lambda x: x * x, a=-10, b=10, start=5, global_max=100, steps=10, plot=False)
    self.run_opt(f=lambda x: np.sin(np.log(np.abs(x))), a=-10, b=10, start=5, global_max=1, steps=10, plot=False)
    self.run_opt(f=lambda x: x / (np.sin(x) + 2), a=-8, b=8, start=3, global_max=4.8, steps=10, plot=False)

  def test_1d_periodic_max_1(self):
    self.run_opt(f=lambda x: x * np.sin(x),
                 a=-10, b=10, start=3,
                 global_max=7.9, delta=0.3,
                 steps=30,
                 plot=False)

  def test_1d_periodic_max_2(self):
    self.run_opt(f=lambda x: x * np.sin(x + 1) / 2,
                 a=-12, b=16, start=3,
                 global_max=6.55, delta=0.5,
                 steps=30,
                 plot=False)

  def test_1d_many_small_peaks(self):
    self.run_opt(f=lambda x: np.exp(np.sin(x * 5) * np.sqrt(x)),
                 a=0, b=10, start=3,
                 global_max=20.3, delta=1.0,
                 steps=30,
                 plot=False)

  # 2-D

  def test_2d_simple(self):
    self.run_opt(f=lambda x: x[0] + x[1],
                 a=(0, 0), b=(10, 10), start=(5, 5),
                 global_max=20, delta=1.0,
                 steps=10,
                 plot=False)

  def test_2d_peak_1(self):
    self.run_opt(f=(lambda x: (x[0] + x[1]) / ((x[0] - 1) ** 2 - np.sin(x[1]) + 2)),
                 a=(0, 0), b=(9, 9), start=None,
                 global_max=8.95,
                 steps=20,
                 plot=False)

  def test_2d_irregular_max_1(self):
    self.run_opt(f=(lambda x: (x[0] + x[1]) / (np.exp(-np.sin(x[0])))),
                 a=(0, 0), b=(9, 9), start=None,
                 global_max=46,
                 steps=20,
                 plot=False)

  def test_2d_irregular_max_2(self):
    self.run_opt(f=lambda x: np.sum(x * np.sin(x + 1) / 2, axis=0),
                 a=(-10, -10), b=(15, 15), start=None,
                 global_max=13.175,
                 steps=40,
                 plot=False)

  def test_2d_periodic_max_1(self):
    self.run_opt(f=lambda x: np.sin(x[0]) + np.cos(x[1]),
                 a=(0, 0), b=(10, 10), start=(5, 5),
                 global_max=2,
                 steps=10,
                 plot=False)

  def test_2d_periodic_max_2(self):
    self.run_opt(f=lambda x: np.sin(x[0]) * np.cos(x[1]),
                 a=(0, 0), b=(10, 10), start=(5, 5),
                 global_max=1,
                 steps=30,
                 plot=False)

  def test_2d_periodic_max_3(self):
    self.run_opt(f=lambda x: np.sin(x[0]) / (np.cos(x[1]) + 2),
                 a=(0, 0), b=(10, 10), start=(5, 5),
                 global_max=1,
                 steps=40,
                 plot=False)

  # 4-D

  def test_4d_simple_1(self):
    self.run_opt(f=lambda x: x[0] + 2*x[1] - x[2] - 2*x[3],
                 a=(-10, -10, -10, -10), b=(10, 10, 10, 10), start=None,
                 global_max=60,
                 steps=50,
                 plot=False)

  def test_4d_simple_2(self):
    self.run_opt(f=lambda x: x[0] + np.sin(x[1]) - x[2] + x[3],
                 a=(-10, -10, -1, -1), b=(10, 10, 1, 1), start=None,
                 global_max=13, delta=1.0,
                 steps=20,
                 plot=False)

  def test_4d_irregular_max(self):
    self.run_opt(f=lambda x: (np.sin(x[0]**2) + np.exp((x[1] - x[2]) / 10)) / (x[3]**2 + 1),
                 a=(-10, -10, -10, -10), b=(10, 10, 10, 10), start=None,
                 global_max=math.e**2 + 1, delta=1.0,
                 steps=50,
                 plot=False)

  # Realistic

  def test_realistic_1(self):
    def f(x):
      init, size, reg, _ = x
      result = 10 * np.cos(size - 3) * np.cos(reg - size / 2)
      result = np.asarray(result)
      result[size > 6] = 10 - size[size > 6]
      result[size < 1] = size[size < 1]
      result[init > 4] = 7 - init[init > 4]
      return result

    self.run_opt(f=f,
                 a=(0, 0, 0, 0), b=(10, 10, 10, 1), start=None,
                 global_max=10, delta=0.5,
                 steps=20)

  # Technical details

  def run_opt(self, f, a, b, start=None, global_max=None, delta=None, steps=10, plot=False):
    if plot:
      self._run(f, a, b, start, steps, batch_size=100000, stop_condition=None)
      self._plot(f, a, b)
      return

    errors = []
    size_list = [1000, 10000, 50000, 100000]
    for batch_size in size_list:
      try:
        delta = delta or abs(global_max) / 10.0
        max_value = self._run(f, a, b, start, steps, batch_size,
                              stop_condition=lambda x: abs(f(x) - global_max) <= delta)
        self.assertAlmostEqual(max_value, global_max, delta=delta)
        return
      except AssertionError as e:
        errors.append(str(e))
        log('Attempt for %d failed: %s' % (batch_size, str(e)))
        if batch_size == size_list[-1]:
          self.fail(',\n                '.join(errors))

  def _run(self, f, a, b, start, steps, batch_size, stop_condition):
    sampler = DefaultSampler()
    sampler.add(lambda: np.random.uniform(a, b))
    self.optimizer = BayesianOptimizer(sampler, mc_batch_size=batch_size)

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

  def _plot(self, f, a, b):
    if type(a) in [tuple, list]:
      if len(a) == 2:
        self._plot_2d(f, a, b)
      else:
        self._scatter_plot_per_dimension()
    else:
      self._plot_1d(f, a, b)

  def _plot_1d(self, f, a, b, grid_size=1000):
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

  def _plot_2d(self, f, a, b, grid_size=200):
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

  def _scatter_plot_per_dimension(self):
    points = self.optimizer.utility.points
    values = self.optimizer.utility.values
    n, d = points.shape

    _, axes = plt.subplots(1, d)
    for j in xrange(d):
      axes[j].scatter(points[:, j], values, s=100, alpha=0.5)
    plt.show()

  def _bar_plot_per_dimension(self):
    points = self.optimizer.utility.points
    values = self.optimizer.utility.values
    n, d = points.shape

    _, axes = plt.subplots(1, d)
    for j in xrange(d):
      p = points[:, j]
      split = np.linspace(np.min(p), np.max(p), 10)
      bar_values = np.zeros((len(split),))
      for k in xrange(len(split) - 1):
        interval = np.logical_and(split[k] < p, p < split[k+1])
        if np.any(interval):
          bar_values[k] = np.mean(values[interval])
      axes[j].bar(split, height=bar_values, width=split[1]-split[0])
    plt.show()

  def _eval_max(self, f, a, b):
    sampler = DefaultSampler()
    sampler.add(lambda: np.random.uniform(a, b))
    batch = sampler.sample(1000000)
    batch = np.swapaxes(batch, 1, 0)
    return np.max(f(batch))

  def _eval_at(self, val):
    log(self.optimizer.utility.compute_values(np.asarray([[val]])))
    log(self.optimizer.utility.compute_values(self.optimizer.utility.points))


if __name__ == "__main__":
  unittest.main()
