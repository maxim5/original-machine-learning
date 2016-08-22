#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import theano
from theano import tensor as T
import numpy as np

trainX = np.linspace(-1, 1, 101)
trueY = 2 * trainX
trainY = trueY + np.random.randn(*trainX.shape) * 0.33      # numpy.ndarray.shape is Tuple of array dimensions.

# symbolic variables
X = T.scalar()
Y = T.scalar()

# linear model
def model(X, w):
    return X * w

# model param init
w = theano.shared(np.asarray(0.0, dtype=theano.config.floatX))  # shared? that's for learned parameters
y = model(X, w)

# cost function
cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)                             # how does this work?!
updates = [[w, w - gradient * 0.01]]

# Compiling to a python function
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(200):
    for x, y in zip(trainX, trainY):
        train(x, y)

print w.get_value()
