#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import unittest

import numpy as np

from artist import Artist
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
                 a=(-5, -5, -5, -5), b=(5, 5, 5, 5), start=None,
                 global_max=16, delta=1.0,
                 steps=20,
                 plot=False)

  def test_4d_irregular_max(self):
    self.run_opt(f=lambda x: (np.sin(x[0]**2) + np.power(2, (x[1] - x[2]) / 5)) / (x[3]**2 + 1),
                 a=(-5, -5, -5, -5), b=(5, 5, 5, 5), start=None,
                 global_max=5, delta=1.0,
                 steps=50,
                 plot=False)

  # 10-D

  def test_10d_simple_1(self):
    self.run_opt(f=lambda x: x[1] + 5 * x[5] - x[7],
                 a=[-1]*10, b=[1]*10, start=None,
                 global_max=7, delta=0.5,
                 steps=30,
                 plot=False)

  def test_10d_simple_2(self):
    self.run_opt(f=lambda x: np.cos(x[0]) + np.sin(x[1]) + np.exp(-x[2]) - np.exp(x[3]),
                 a=[-1]*10, b=[1]*10, start=None,
                 global_max=4.2, delta=0.3,
                 steps=30,
                 plot=False)

  def test_10d_simple_3(self):
    self.run_opt(f=lambda x: (x[0] * x[1] - x[2] * x[3]) * x[4],
                 a=[0]*10, b=[1]*10, start=None,
                 global_max=1, delta=0.2,
                 steps=30,
                 plot=False)

  def test_10d_simple_4(self):
    self.run_opt(f=lambda x: x[0]*x[1]*x[2]*x[3]*x[4]*x[5],
                 a=[0]*10, b=[1]*10, start=None,
                 global_max=1, delta=0.5,
                 steps=50,
                 plot=False)

  def test_10d_simple_5(self):
    self.run_opt(f=lambda x: np.sum(x, axis=0),
                 a=[0]*10, b=[1]*10, start=None,
                 global_max=10, delta=1.5,
                 steps=30,
                 plot=False)

  def test_10d_irregular_max(self):
    self.run_opt(f=lambda x: (np.sin(x[0]**2) + np.power(2, (x[1] - x[2]))) / (x[3]**2 + 1),
                 a=[0]*10, b=[1]*10, start=None,
                 global_max=2.8, delta=0.2,
                 steps=30,
                 plot=False)

  # Realistic

  def test_realistic_4d(self):
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

  def test_realistic_10d(self):
    def f(x):
      init, num1, size1, activation1, dropout1, num2, size2, activation2, dropout2, fc_size = x
      result = np.sin(num1 * size1 + activation1) + np.cos(num2 * size2 + activation2) + fc_size
      result = np.asarray(result)
      result[size1 > 0.5] = 1 - size1[size1 > 0.5]
      result[size2 > 0.5] = 1 - size1[size2 > 0.5]
      result[dropout1 < 0.3] = dropout1[dropout1 < 0.3]
      result[dropout2 < 0.4] = dropout1[dropout2 < 0.4]
      result[init > 0.5] = np.exp(-init[init > 0.5])
      return result

    self.run_opt(f=f,
                 a=[0]*10, b=[1]*10, start=None,
                 global_max=3, delta=0.3,
                 steps=50)

  # Technical details

  def run_opt(self, f, a, b, start=None, global_max=None, delta=None, steps=10, plot=False):
    if plot:
      self._run(f, a, b, start, steps, batch_size=100000, stop_condition=None)
      self._plot(f, a, b)
      return

    errors = []
    size_list = [1000, 10000, 50000, 100000]

    for batch_size in size_list:
      delta = delta or abs(global_max) / 10.0
      max_value = self._run(f, a, b, start, steps, batch_size,
                            stop_condition=lambda x: abs(f(x) - global_max) <= delta)
      if abs(max_value - global_max) <= delta:
        return
      msg = 'Failure %6d: max=%.3f, expected=%.3f within delta=%.3f' % (batch_size, max_value, global_max, delta)
      errors.append(msg)

    log('\n                      '.join(errors))
    self.fail('\n                '.join(errors))

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
    artist = Artist(optimizer=self.optimizer)
    if type(a) in [tuple, list]:
      if len(a) == 2:
        artist.plot_2d(f, a, b)
      else:
        artist.scatter_plot_per_dimension()
    else:
      artist.plot_1d(f, a, b)

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
