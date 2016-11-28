#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import numpy as np
import tflearn
import tensorflow as tf

data = [
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


N = 5

X = np.array([item[0][:N] for item in data])
Y = np.array([[item[1]] for item in data])

TEST = [
  ([43, 51, 55, 57, 62, 64, 67, 66, 68, 68, 71], 71),
  ([42, 50, 53, 59, 61, 63, 64, 66, 67, 68, 70], 70),
  ([39, 46, 51, 54, 59, 61, 62, 63, 65, 65, 66], 66),
  ([41, 51, 52, 56, 61, 62, 64, 66, 68, 68, 69], 69),
  ([44, 48, 54, 57, 60, 62, 65, 65, 66, 66, 68], 68),
  ([40, 51, 54, 58, 62, 66, 68, 69, 71, 70, 72], 72),
]
TEST_X = np.array([item[0][:N] for item in TEST])
TEST_Y = np.array([[item[1]] for item in TEST])

with tf.Graph().as_default():
  g = tflearn.input_data(shape=[None, N])
  g = tflearn.fully_connected(g, 1, activation='linear')
  g = tflearn.regression(g, optimizer='sgd', learning_rate=0.001, loss='mean_square')

  model = tflearn.DNN(g)
  model.fit(X, Y, n_epoch=1000, snapshot_epoch=False)

W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
print W.shape

def predict_y(x, y):
  predict = model.predict(x)
  predict = predict[0][0]
  print '[1] predict=%.3f true=%.3f error=%.3f' % (predict, y, abs(predict-y))

  predict = x.dot(W)
  print '[2] predict=%.3f true=%.3f error=%.3f' % (predict, y, abs(predict-y))

print '\nChecking training data:'
for i in xrange(X.shape[0]):
  x = X[i].reshape(1, -1)
  y = Y[i][0]
  predict_y(x, y)


print '\nChecking test data:'
for i in xrange(TEST_X.shape[0]):
  x = TEST_X[i].reshape(1, -1)
  y = TEST_Y[i][0]
  predict_y(x, y)
