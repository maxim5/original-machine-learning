#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf


class ConvModel:
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.x = tf.placeholder(tf.float32, [None, input_shape[0] * input_shape[1] * input_shape[2]])
    self.y = tf.placeholder(tf.float32, [None, num_classes])


  def conv2d_relu(self, image, W, b, strides):
    layer = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, b)
    layer = tf.nn.relu(layer)
    return layer


  def conv_layer(self, image, filter_size, pool_size, dropout):
    conv = image
    for filter in filter_size:
      W = tf.Variable(tf.random_normal(filter))
      b = tf.Variable(tf.random_normal(filter[-1:]))
      conv = self.conv2d_relu(conv, W, b, strides=1)

    layer = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding='SAME')
    layer = tf.nn.dropout(layer, keep_prob=dropout)
    return layer


  def fully_connected_layer(self, features, shape, dropout):
    W = tf.Variable(tf.random_normal(shape))
    b = tf.Variable(tf.random_normal(shape[-1:]))
    layer = tf.reshape(features, [-1, W.get_shape().as_list()[0]])
    layer = tf.add(tf.matmul(layer, W), b)
    layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, dropout)
    return layer


  def output_layer(self, input, shape):
    W_out = tf.Variable(tf.random_normal(shape))
    b_out = tf.Variable(tf.random_normal(shape[-1:]))
    layer = tf.add(tf.matmul(input, W_out), b_out)
    return layer


  def conv_net(self, **hyper_params):
    self.dropout_conv = tf.placeholder(tf.float32)
    self.dropout_fc = tf.placeholder(tf.float32)

    ch = self.input_shape[-1]
    layer0 = tf.reshape(self.x, shape=(-1,)+self.input_shape)
    layer1 = self.conv_layer(layer0, filter_size=([3, 3, ch,  16], [3, 3, 16,  32]), pool_size=[1, 2, 2, 1], dropout=self.dropout_conv)
    layer2 = self.conv_layer(layer1, filter_size=([3, 3, 32,  32], [3, 3, 32,  64]), pool_size=[1, 2, 2, 1], dropout=self.dropout_conv)
    layer3 = self.conv_layer(layer2, filter_size=([3, 3, 64,  64], [3, 3, 64, 128]), pool_size=[1, 2, 2, 1], dropout=self.dropout_conv)

    reduced_size = 4  # 28 / 8
    layer_fc = self.fully_connected_layer(layer3, shape=[reduced_size * reduced_size * 128, 1024], dropout=self.dropout_fc)
    layer_out = self.output_layer(layer_fc, shape=[1024, self.num_classes])

    return layer_out


  def build_graph(self, **hyper_params):
    prediction = self.conv_net(**hyper_params)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))
    optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params['learning_rate']).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    return optimizer, cost, accuracy, init


  def feed_dict(self, data_set=None, images=None, labels=None, **hyper_params):
    if images is None and data_set is not None:
      images = data_set.images
    if labels is None and data_set is not None:
      labels = data_set.labels
    return {
      self.x: images,
      self.y: labels,
      self.dropout_conv: hyper_params.get('dropout_conv', 1.0),
      self.dropout_fc: hyper_params.get('dropout_fc', 1.0),
    }
