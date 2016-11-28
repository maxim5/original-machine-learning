#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np

class DataSet(object):
  def __init__(self, x, y):
    x = np.array(x)
    y = np.array(y)
    assert x.shape[0] == y.shape[0]

    self.size = x.shape[0]
    self.x = x
    self.y = y
    self.step = 0
    self.epochs_completed = 0
    self.index_in_epoch = 0
    self.just_completed = False

  @property
  def index(self):
    return self.epochs_completed * self.size + self.index_in_epoch

  def reset_counters(self):
    self.step = 0
    self.epochs_completed = 0
    self.index_in_epoch = 0

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    if self.just_completed:
      permutation = np.arange(self.size)
      np.random.shuffle(permutation)
      self.x = self.x[permutation]
      self.y = self.y[permutation]

    self.step += 1
    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    end = min(self.index_in_epoch, self.size)
    if self.index_in_epoch >= self.size:
      self.index_in_epoch = 0
    self.just_completed = end == self.size
    self.epochs_completed += int(self.just_completed)
    return self.x[start:end], self.y[start:end]


def merge_data_sets(ds1, ds2):
  x = np.concatenate([ds1.x, ds2.x], axis=0)
  y = np.concatenate([ds1.y, ds2.y], axis=0)
  return DataSet(x, y)


class Data(object):
  def __init__(self, train, validation, test):
    self.train = train
    self.validation = validation
    self.test = test

  def reset_counters(self):
    self.train.reset_counters()
    self.validation.reset_counters()
    self.test.reset_counters()

  def merge_validation_to_train(self):
    self.train = merge_data_sets(self.train, self.validation)
    self.validation = self.test
