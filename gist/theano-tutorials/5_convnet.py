#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

# A convolutional NN (2012+)
# THEANO_FLAGS='floatX=float32'

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np

import load

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.0)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.0)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
    # conv -> activate -> pool -> dropout

    l1a = rectify(conv2d(X, w, border_mode="full"))
    l1 = max_pool_2d(l1a, (2, 2), ignore_border=False)
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2), ignore_border=False)
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2), ignore_border=False)
    l3 = T.flatten(l3b, outdim=2)   # convert from 4tensor to normal matrix
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    py_x = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, py_x

trainX, testX, trainY, testY = load.mnist(onehot=True)

trainX = trainX.reshape(-1, 1, 28, 28)  # reshape into conv 4 tensor (b, c, 0, 1) format
testX = testX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))         # conv weights (n_kernels, n_channels, kernel_w, kernel_h)
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))   # highest conv later has 128 filter and 3x3 grid of responses
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5)     # dropout during training
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0.0, 0.0)                                   # clean for prediction
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, learning_rate=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

def mini_batch(total, size):
    return zip(range(0, total, size),
               range(size, total, size))

print "Start"
for epoch in range(100):
    for start, end in mini_batch(len(trainX), 128):
        cost = train(trainX[start:end], trainY[start:end])
    print epoch, np.mean(np.argmax(testY, axis=1) == predict(testX))
