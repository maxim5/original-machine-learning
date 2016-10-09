#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime

import numpy as np

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import load


########################################################################################################################
# General stuff
########################################################################################################################


def log(*msg):
    all = (datetime.datetime.now(), ) + msg
    print ' '.join([str(it) for it in all])


def get_data():
    trainX, testX, trainY, testY = load.mnist(onehot=True)
    trainX = trainX.reshape(-1, 1, 28, 28)  # reshape into conv 4 tensor (b, c, 0, 1) format
    testX = testX.reshape(-1, 1, 28, 28)
    return trainX, testX, trainY, testY


def mini_batch(total, size):
    return zip(range(0, total, size),
               range(size, total, size))


def run(provider):
    trainX, testX, trainY, testY = get_data()
    train, predict = provider()

    log("Start train")
    for epoch in range(100):
        for start, end in mini_batch(len(trainX), 128):
            train(trainX[start:end], trainY[start:end])
        log(epoch, np.mean(np.argmax(testY, axis=1) == predict(testX)))


########################################################################################################################
# Machine learning util
########################################################################################################################


def init_weights(shape):
    X = np.random.randn(*shape) * 0.01
    as_float = np.asarray(X, dtype=theano.config.floatX)
    return theano.shared(as_float)


def rectify(X):
    return T.maximum(X, 0.0)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p):
    global srng
    srng = RandomStreams()
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
