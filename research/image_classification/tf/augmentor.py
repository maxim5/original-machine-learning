#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
import skimage.transform
import skimage.util
import random
from tflearn import ImageAugmentation


class MyImageAugmentation(ImageAugmentation):
  def add_random_scale(self, max_scale_x=1.0, max_scale_y=1.0):
    assert 0.0 < max_scale_x <= 1.0
    assert 0.0 < max_scale_y <= 1.0
    if max_scale_x < 1.0 or max_scale_y < 1.0:
      self.methods.append(self._random_scale)
      self.args.append([max_scale_x, max_scale_y])

  def _random_scale(self, batch, max_scale_x, max_scale_y):
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        scale_x = np.random.uniform(max_scale_x, 1.0)
        scale_y = np.random.uniform(max_scale_y, 1.0)
        scaled = skimage.transform.rescale(batch[i], scale=(scale_x, scale_y), preserve_range=True)
        batch[i] = pad_to_shape(scaled, batch[i].shape)
    return batch


def pad_to_shape(image, target_shape):
  current_shape = image.shape
  diff = (target_shape[0] - current_shape[0], target_shape[1] - current_shape[1])
  pad = (diff[0] / 2, diff[1] / 2)
  padding = ((pad[0], diff[0]-pad[0]), (pad[1], diff[1]-pad[1]), (0, 0))
  return skimage.util.pad(image, pad_width=padding, mode='constant')
