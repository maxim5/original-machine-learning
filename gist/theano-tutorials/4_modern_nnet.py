#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

# A "modern" deep NN as of 2012+
# Changes: sigmoid -> ReLu (aka rectify)
#          dropout (a form of noise between layers)
#          2 hidden layers instead of 1
# Hits the accuracy of 99%

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

import load

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.0)

# Numerically stable softmax (ordinary T.nnet.softmax can explode because of exponent)
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        # This code is a trick to initialize `acc` to the same shape as the parameters except all zeros
        acc = theano.shared(p.get_value() * 0.0)                # a running average of the magnitude of the gradient
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates

def dropout(X, p=0.0):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)     # randomly drop
        X /= retain_prob                                                           # account for bias
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

trainX, testX, trainY, testY = load.mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

hidden_layer_size = 625
w_h = init_weights(shape=(trainX.shape[1], hidden_layer_size))
w_h2 = init_weights(shape=(hidden_layer_size, hidden_layer_size))
w_o = init_weights(shape=(hidden_layer_size, trainY.shape[1]))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
# y_prediction = T.argmax(noise_py_x, axis=1)

h, h2, py_x = model(X, w_h, w_h2, w_o, 0.0, 0.0)
y_prediction = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, learning_rate=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_prediction, allow_input_downcast=True)

def mini_batch(total, size):
    return zip(range(0, total, size),
               range(size, total, size))

print "Start"
for epoch in range(100):
    for start, end in mini_batch(len(trainX), 128):
        cost = train(trainX[start:end], trainY[start:end])
    print epoch, np.mean(np.argmax(testY, axis=1) == predict(testX))
