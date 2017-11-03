#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import numpy as np

def conv_forward_naive(x, w, b, stride, pad):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - 'stride': The number of pixels between adjacent receptive fields in the
    horizontal and vertical directions.
  - 'pad': The number of pixels that will be used to zero-pad the input.
  """
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, H_out, W_out))

  x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

  for n in xrange(N):
    for f in xrange(F):
      filter_w = w[f, :, :, :]                            # 3-dimensional: (C, HH, WW)
      for out_i, i in enumerate(xrange(0, H, stride)):
        for out_j, j in enumerate(xrange(0, W, stride)):
          image_patch = x_pad[n, :, i:i+HH, j:j+WW]       # 3-dimensional: (C, HH, WW)
          out[n, f, out_i, out_j] += np.sum(filter_w * image_patch)
      out[n, f, :, :] += b[f]

  return out
