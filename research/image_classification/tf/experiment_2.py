#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from common import *
from conv_model import ConvModel


def run(data_sets):
  model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  train(data_sets=data_sets, model=model, epochs=10, batch_size=128, learning_rate=0.001, dropout_conv=0.8, dropout_fc=0.8)


def hyper_tune(data_sets):
  best_accuracy = 0
  while True:
    tf.reset_default_graph()
    model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
    learning_rate = 10**np.random.uniform(-3, -5)
    dropout_conv = np.random.uniform(0.7, 1.0)
    dropout_fc = np.random.uniform(0.7, 1.0)
    accuracy = train(data_sets=data_sets, model=model, epochs=0, batch_size=128, learning_rate=learning_rate, dropout_conv=dropout_conv, dropout_fc=dropout_fc)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      log("! new best_acc=%.4f" %  best_accuracy)


if __name__ == "__main__":
  mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  hyper_tune(mnist)
