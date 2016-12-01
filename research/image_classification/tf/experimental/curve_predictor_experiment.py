#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
import tflearn
import tensorflow as tf

TRAIN = [
  ([40, 50, 55, 60, 63, 65, 68, 67, 69, 70, 72], 72),
  ([30, 34, 40, 42, 46, 46, 50, 52, 53, 55, 54], 55),
  ([32, 40, 45, 48, 54, 56, 57, 58, 60, 61, 61], 61),
  ([37, 45, 51, 54, 57, 60, 59, 63, 64, 67, 68], 68),
  ([44, 48, 53, 59, 63, 63, 64, 64, 68, 69, 70], 70),
  ([42, 49, 53, 57, 60, 63, 62, 65, 66, 68, 68], 68),
  ([41, 47, 55, 58, 61, 62, 65, 67, 69, 71, 71], 71),
  ([36, 43, 50, 52, 57, 61, 61, 62, 64, 66, 69], 69),
  ([20, 25, 30, 34, 38, 42, 46, 50, 53, 55, 54], 55),
  ([25, 27, 35, 40, 43, 47, 51, 52, 55, 56, 57], 57),
  ([29, 34, 40, 45, 50, 53, 56, 57, 57, 60, 59], 60),
  ([10, 16, 25, 30, 32, 35, 36, 35, 37, 37, 37], 37),
  ([41, 51, 52, 56, 61, 62, 65, 66, 68, 67, 69], 69),
  ([45, 48, 54, 57, 60, 62, 62, 65, 65, 66, 66], 66),
  ([22, 30, 33, 37, 39, 40, 45, 47, 48, 48, 51], 51),
  ([42, 51, 55, 56, 58, 61, 62, 64, 65, 66, 68], 68),
  ([34, 43, 45, 50, 52, 55, 61, 61, 60, 66, 64], 66),
  ([20, 24, 25, 30, 31, 33, 36, 37, 40, 42, 43], 43),
  ([44, 51, 55, 58, 60, 63, 65, 64, 66, 67, 67], 67),
  ([41, 49, 55, 58, 61, 61, 66, 67, 68, 68, 71], 71),
  ([34, 45, 48, 53, 55, 57, 61, 62, 63, 63, 64], 64),
]
TEST = [
  ([43, 51, 55, 57, 62, 64, 67, 66, 68, 68, 71], 71),
  ([42, 50, 53, 59, 61, 63, 64, 66, 67, 68, 70], 70),
  ([39, 46, 51, 54, 59, 61, 62, 63, 65, 65, 66], 66),
  ([41, 51, 52, 56, 61, 62, 64, 66, 68, 68, 69], 69),
  ([44, 48, 54, 57, 60, 62, 65, 65, 66, 66, 68], 68),
  ([40, 51, 54, 58, 62, 66, 68, 69, 71, 70, 72], 72),
]

N = 5
X = np.array([item[0][:N] for item in TRAIN])
Y = np.array([[item[1]] for item in TRAIN])
TEST_X = np.array([item[0][:N] for item in TEST])
TEST_Y = np.array([[item[1]] for item in TEST])

########################################################################################################################
# Models
########################################################################################################################

class Predictor:
  def __init__(self):
    self.error = 0

  def prepare(self):
    pass

  def predict(self, x):
    pass

  def describe(self):
    return self.__class__.__name__

class TfLinear(Predictor):
  def __init__(self):
    Predictor.__init__(self)
    with tf.Graph().as_default():
      g = tflearn.input_data(shape=[None, N])
      g = tflearn.fully_connected(g, 1, activation='linear')
      g = tflearn.regression(g, optimizer='sgd', learning_rate=0.001, loss='mean_square')
      self.model = tflearn.DNN(g)
      self.model.fit(X, Y, n_epoch=1000, snapshot_epoch=False)

  def predict(self, x):
    return self.model.predict(x)[0][0]

class SimpleLinear(Predictor):
  def __init__(self):
    Predictor.__init__(self)
    self.W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)

  def predict(self, x):
    return x.dot(self.W)

class BiasLinear(Predictor):
  def __init__(self):
    Predictor.__init__(self)
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    self.W = np.linalg.pinv(X1.T.dot(X1)).dot(X1.T).dot(Y)

  def predict(self, x):
    x1 = np.insert(x, 0, 1)
    return x1.dot(self.W)

models = [SimpleLinear(), BiasLinear(), TfLinear()]

# L-BFGS
# http://stats.stackexchange.com/questions/17436/logistic-regression-with-lbfgs-solver
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
# http://ai.stanford.edu/~quocle/LeNgiCoaLahProNg11.pdf

########################################################################################################################
# Run
########################################################################################################################

def predict_y(model, x, y):
  predict = model.predict(x)
  error = abs(predict - y)
  model.error += error
  print '[%12s] predict=%.3f true=%.3f error=%.3f' % (model.describe(), predict, y, error)
  return error

def check(data):
  for model in models:
    model.error = 0
    for x, y in data:
      predict_y(model, x, y)
    print '[%12s] Total error: %.4f' % (model.describe(), model.error)
    print

training = [(X[i].reshape(1, -1), Y[i][0]) for i in xrange(X.shape[0])]
testing =  [(TEST_X[i].reshape(1, -1), TEST_Y[i][0]) for i in xrange(TEST_X.shape[0])]

print '\nChecking training data:'
check(training)

print '\nChecking test data:'
check(testing)
