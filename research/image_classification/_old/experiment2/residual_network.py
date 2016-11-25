#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf
tf.python.control_flow_ops = tf

import tflearn
import tflearn.data_utils as du

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
data = mnist.read_data_sets("/home/maxim/p/dat/mnist-tf", one_hot=True)
X, Y, valX, valY, testX, testY = data.train.images, data.train.labels, \
                                 data.validation.images, data.validation.labels, \
                                 data.test.images, data.test.labels

X = X.reshape([-1, 28, 28, 1])
valX = valX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
X, mean = du.featurewise_zero_center(X)
valX = du.featurewise_zero_center(valX, mean)
testX = du.featurewise_zero_center(testX, mean)

# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 1])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist', max_checkpoints=None, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, validation_set=(valX, valY), show_metric=True, batch_size=512, run_id='resnet_mnist')
print model.evaluate(testX, testY, batch_size=128)
