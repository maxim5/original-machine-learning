#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

from tensorflow.examples.tutorials.mnist import input_data

from common import train
from conv_model import ConvModel


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  train(data_sets=mnist, model=model, epochs=10, batch_size=128, learning_rate=0.001, dropout_conv=0.8, dropout_fc=0.8)
