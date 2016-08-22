#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

# A NN as of 2000s
# Hits the accuracy of 98.5%

import theano
from theano import tensor as T
import numpy as np

import load

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    py_x = T.nnet.softmax(T.dot(h, w_o))
    return py_x

def sgd(cost, params, learning_rate=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * learning_rate])
    return updates

trainX, testX, trainY, testY = load.mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

hidden_layer_size = 625
w_h = init_weights(shape=(trainX.shape[1], hidden_layer_size))
w_o = init_weights(shape=(hidden_layer_size, trainY.shape[1]))

py_x = model(X, w_h, w_o)
y_prediction = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_prediction, allow_input_downcast=True)

def mini_batch(total, size):
    return zip(range(0, total, size),
               range(size, total, size))

for epoch in range(100):
    for start, end in mini_batch(len(trainX), 128):
        cost = train(trainX[start:end], trainY[start:end])
    print epoch, np.mean(np.argmax(testY, axis=1) == predict(testX))
