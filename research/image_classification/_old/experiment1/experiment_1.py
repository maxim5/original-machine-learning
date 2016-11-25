#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from util import *

import theano
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


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
    l3 = T.flatten(l3b, outdim=2)  # convert from 4tensor to normal matrix
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    py_x = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, py_x


def compile_graph():
    log("Compile")

    X = T.ftensor4()
    Y = T.fmatrix()
    w = init_weights((32, 1, 3, 3))         # conv weights (n_kernels, n_channels, kernel_w, kernel_h)
    w2 = init_weights((64, 32, 3, 3))
    w3 = init_weights((128, 64, 3, 3))
    w4 = init_weights((128 * 3 * 3, 625))   # highest conv later has 128 filter and 3x3 grid of responses
    w_o = init_weights((625, 10))

    _, _, _, _, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5)     # dropout during training
    _, _, _, _, py_x       = model(X, w, w2, w3, w4, w_o, 0.0, 0.0)     # clean for prediction
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w, w2, w3, w4, w_o]
    updates = RMSprop(cost, params, learning_rate=0.001)

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    return train, predict


if __name__ == "__main__":
    run(compile_graph)
