#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

class Artist(object):
  def __init__(self, *args, **kwargs):
    super(Artist, self).__init__()
    if len(args) == 3:
      self.points, self.values, self.utility = args
    else:
      strategy = kwargs.get('strategy')
      if strategy is not None:
        self.points = strategy.points
        self.values = strategy.values
        self.utility = strategy._method
      self.names = kwargs.get('names', {})

  def plot_1d(self, f, a, b, grid_size=1000):
    grid = np.linspace(a, b, num=grid_size).reshape((-1, 1))
    mu, sigma = self.utility.mean_and_std(grid)

    plt.plot(grid, f(grid), color='black', linewidth=1.5, label='f')
    plt.plot(grid, mu, color='red', label='mu')
    plt.plot(grid, mu + sigma, color='blue', linewidth=0.4, label='mu+sigma')
    plt.plot(grid, mu - sigma, color='blue', linewidth=0.4)
    plt.plot(self.points, f(np.asarray(self.points)), 'o', color='red')
    plt.xlim([a - 0.5, b + 0.5])
    # plt.legend()
    plt.show()

  def plot_2d(self, f, a, b, grid_size=200):
    grid_x = np.linspace(a[0], b[0], num=grid_size).reshape((-1, 1))
    grid_y = np.linspace(a[1], b[1], num=grid_size).reshape((-1, 1))
    x, y = np.meshgrid(grid_x, grid_y)

    merged = np.stack([x.flatten(), y.flatten()])
    z = f(merged).reshape(x.shape)

    swap = np.swapaxes(merged, 0, 1)
    mu, sigma = self.utility.mean_and_std(swap)
    mu = mu.reshape(x.shape)
    sigma = sigma.reshape(x.shape)

    points = np.asarray(self.points)
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

  def scatter_plot_per_dimension(self):
    points = np.array(self.points)
    values = np.array(self.values)
    n, d = points.shape
    rows = int(math.sqrt(d))
    cols = (d + rows - 1) / rows

    _, axes = plt.subplots(rows, cols)
    axes = axes.reshape(-1)
    for j in xrange(d):
      axes[j].scatter(points[:, j], values, s=100, alpha=0.5)
      axes[j].set_title(self.names.get(j, str(j)))
    plt.show()

  def bar_plot_per_dimension(self):
    points = np.array(self.points)
    values = np.array(self.values)
    n, d = points.shape
    rows = int(math.sqrt(d))
    cols = (d + rows - 1) / rows

    _, axes = plt.subplots(rows, cols)
    axes = axes.reshape(-1)
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
