#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
from image_classification.tf.data_set import DataSet

import unittest


class DataSetTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    super(DataSetTest, cls).setUpClass()
    np.random.shuffle = lambda x : x


  def test_fit(self):
    arr = np.asanyarray([1, 2, 3, 4])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2], 2, 2, 0, False)
    self.check_next_batch(2, [3, 4], 4, 0, 1, True)
    self.check_next_batch(2, [1, 2], 6, 2, 1, False)
    self.check_next_batch(2, [3, 4], 8, 0, 2, True)


  def test_fit2(self):
    arr = np.asanyarray([1, 2, 3, 4, 5, 6])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2],  2, 2, 0, False)
    self.check_next_batch(2, [3, 4],  4, 4, 0, False)
    self.check_next_batch(2, [5, 6],  6, 0, 1, True)
    self.check_next_batch(2, [1, 2],  8, 2, 1, False)
    self.check_next_batch(2, [3, 4], 10, 4, 1, False)
    self.check_next_batch(2, [5, 6], 12, 0, 2, True)


  def test_does_not_fit(self):
    arr = np.asanyarray([1, 2, 3, 4, 5])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(2, [1, 2],  2, 2, 0, False)
    self.check_next_batch(2, [3, 4],  4, 4, 0, False)
    self.check_next_batch(2, [5,  ],  5, 0, 1, True)
    self.check_next_batch(2, [1, 2],  7, 2, 1, False)
    self.check_next_batch(2, [3, 4],  9, 4, 1, False)
    self.check_next_batch(2, [5,  ], 10, 0, 2, True)


  def test_too_small_batch(self):
    arr = np.asanyarray([1, 2, 3])
    self.ds = DataSet(arr, arr)
    self.check_next_batch(4, [1, 2, 3], 3, 0, 1, True)
    self.check_next_batch(4, [1, 2, 3], 6, 0, 2, True)
    self.check_next_batch(4, [1, 2, 3], 9, 0, 3, True)


  def check_next_batch(self, batch_size, array, index, index_in_epoch, epochs_completed, just_completed):
    batch = self.ds.next_batch(batch_size)
    self.assertEquals(list(batch[0]), array)
    self.assertEquals(list(batch[1]), array)
    self.assertEquals(self.ds.index, index)
    self.assertEquals(self.ds.index_in_epoch, index_in_epoch)
    self.assertEquals(self.ds.epochs_completed, epochs_completed)
    self.assertEquals(self.ds.just_completed, just_completed)


if __name__ == '__main__':
  unittest.main()
