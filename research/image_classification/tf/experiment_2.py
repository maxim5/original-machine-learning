#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

from tensorflow.examples.tutorials.mnist import input_data

from common import train
from conv_model import ConvModel


if __name__ == "__main__":
  model = ConvModel()
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  train(data_sets=mnist, model=model, epochs=10, batch_size=128, learning_rate=0.001, dropout_conv=0.8, dropout_fc=0.8)
