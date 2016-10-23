#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np


class DataSet(object):
  def __init__(self, x, y):
    assert x.shape[0] == y.shape[0]
    self.size = x.shape[0]
    self.x = x
    self.y = y
    self.epochs_completed = 0
    self.index_in_epoch = 0


  def reset_counters(self):
    self.epochs_completed = 0
    self.index_in_epoch = 0


  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size <= self.size

    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    if self.index_in_epoch > self.size:
      self.epochs_completed += 1

      permutation = np.arange(self.size)
      np.random.shuffle(permutation)
      self.x = self.x[permutation]
      self.y = self.y[permutation]

      start = 0
      self.index_in_epoch = batch_size

    end = self.index_in_epoch
    return self.x[start:end], self.y[start:end]


class Data(object):
  def __init__(self, train, validation, test):
    self.train = train
    self.validation = validation
    self.test = test


  def reset_counters(self):
    self.train.reset_counters()
    self.validation.reset_counters()
    self.test.reset_counters()
