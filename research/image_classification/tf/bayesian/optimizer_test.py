#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from optimizer import BayesianOptimizer
from sampler import DefaultSampler

from image_classification.tf.log import log


def test_simple_1d_1():
  simple_1d(a=-10, b=10, start=5, f=lambda x: np.abs(np.sin(x) / x), steps=10, plot=True)

def test_simple_1d_2():
  simple_1d(a=-10, b=10, start=5, f=lambda x: x * x, steps=10, plot=True)

def test_simple_1d_3():
  simple_1d(a=-10, b=10, start=5, f=lambda x: np.sin(np.log(np.abs(x))), steps=10, plot=True)

def test_simple_1d_4():
  simple_1d(a=-10, b=10, start=3, f=lambda x: x * np.sin(x), steps=40, plot=True)

def test_simple_1d_5():
  simple_1d(a=-8,  b=8,  start=3, f=lambda x: x / (np.sin(x)+2), steps=10, plot=True)

def test_simple_1d_6():
  simple_1d(a=-12, b=16,  start=3, f=lambda x: x * np.sin(x+1) / 2, steps=40, plot=True)

def test_simple_1d_7():
  simple_1d(a=0,   b=10,  start=3, f=lambda x: np.exp(np.sin(x*5)*np.sqrt(x)), steps=40, plot=True)


def test_simple_2d_1():
  simple_2d(a=(0, 0),  b=(10, 10), start=(5, 5), f=lambda x: np.sum(x, axis=0), steps=10, plot=True)

def test_simple_2d_2():
  simple_2d(a=(0, 0),  b=(10, 10), start=(5, 5), f=lambda x: np.sin(x[0]) + np.cos(x[1]), steps=10, plot=True)


def simple_1d(a, b, start, f, steps=10, plot=True):
  optimizer = run(a, b, f, start, steps)
  if plot:
    plot_1d(a, b, f, optimizer)


def simple_2d(a, b, start, f, steps=10, plot=True):
  optimizer = run(a, b, f, start, steps)
  if plot:
    plot_2d(a, b, f, optimizer)


def run(a, b, f, start=None, steps=10):
  sampler = DefaultSampler()
  sampler.add(lambda : np.random.uniform(a, b))

  optimizer = BayesianOptimizer(sampler)

  def add(x):
    optimizer.add_point(np.asarray(x), f(x))

  if start is not None:
    add(start)
  for i in xrange(steps):
    x = optimizer.next_proposal()
    log('selected_point=%s -> true=%.6f' % (x, f(x)))
    add(x)

  i = np.argmax(optimizer.values)
  log('Best found: %s -> %.6f' % (optimizer.points[i], optimizer.values[i]))

  return optimizer


def plot_1d(a, b, f, optimizer, grid_size=1000):
  grid = np.linspace(a, b, num=grid_size).reshape((-1, 1))
  mu, sigma = optimizer.utility.mean_and_std(grid)

  plt.plot(grid, f(grid), color='black', linewidth=1.5, label='f')
  plt.plot(grid, mu, color='red', label='mu')
  plt.plot(grid, mu + sigma, color='blue', linewidth=0.4, label='mu+sigma')
  plt.plot(grid, mu - sigma, color='blue', linewidth=0.4)
  plt.plot(optimizer.points, f(np.asarray(optimizer.points)), 'o', color='red')
  plt.xlim([a-0.5, b+0.5])
  #plt.legend()
  plt.show()


def plot_2d(a, b, f, optimizer, grid_size=200):
  grid_x = np.linspace(a[0], b[0], num=grid_size).reshape((-1, 1))
  grid_y = np.linspace(a[1], b[1], num=grid_size).reshape((-1, 1))
  x, y = np.meshgrid(grid_x, grid_y)

  merged = np.stack([x.flatten(), y.flatten()])
  z = f(merged).reshape(x.shape)

  swap = np.swapaxes(merged, 0, 1)
  mu, sigma = optimizer.utility.mean_and_std(swap)
  mu = mu.reshape(x.shape)
  sigma = sigma.reshape(x.shape)

  points = np.asarray(optimizer.points)
  xs = points[:,0]
  ys = points[:,1]
  zs = f(np.swapaxes(points, 0, 1))

  fig = plt.figure()
  ax = Axes3D(fig)
  ax.plot_surface(x, y, z, color='black', label='f', alpha=0.7,
                  linewidth=0, antialiased=False)
  ax.plot_surface(x, y, mu, color='red', label='mu', alpha=0.5)
  ax.plot_surface(x, y, mu + sigma, color='blue', label='mu+sigma', alpha=0.3)
  ax.plot_surface(x, y, mu - sigma, color='blue', alpha=0.3)
  ax.scatter(xs, ys, zs, color='red', marker='o', s=100)
  #plt.legend()
  plt.show()


def debug(optimizer, val):
  log(optimizer.utility.compute_values(np.asarray([[val]])))
  log(optimizer.utility.compute_values(optimizer.utility.points))


if __name__ == "__main__":
  test_simple_2d_2()
