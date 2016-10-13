#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf
from common import zip_longest


class ConvModel:
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes


  def init(self, shape):
    return tf.random_normal(shape) * self.hyper_params['init_stdev']


  def conv2d_relu(self, image, W, b, strides):
    layer = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, b)
    layer = tf.nn.relu(layer)
    return layer


  def conv_layer(self, image, filter_size, pool_size, dropout):
    conv = image
    for filter in filter_size:
      W = tf.Variable(self.init(filter))
      b = tf.Variable(self.init(filter[-1:]))
      conv = self.conv2d_relu(conv, W, b, strides=1)

    layer = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding='SAME')
    layer = tf.nn.dropout(layer, keep_prob=dropout)
    return layer


  def reduce_layer(self, input):
    input_shape = input.get_shape()
    layer = tf.nn.avg_pool(input, ksize=[1, input_shape[1].value, input_shape[2].value, 1], strides=[1, 1, 1, 1], padding='VALID')
    return layer


  def fully_connected_layer(self, input, size, dropout):
    input_shape = input.get_shape()
    fc_shape = [input_shape[1].value * input_shape[2].value * input_shape[3].value, size]
    W = tf.Variable(self.init(fc_shape))
    b = tf.Variable(self.init(fc_shape[-1:]))
    layer = tf.reshape(input, [-1, W.get_shape().as_list()[0]])
    layer = tf.add(tf.matmul(layer, W), b)
    layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, dropout)
    return layer


  def output_layer(self, input, shape):
    W_out = tf.Variable(self.init(shape))
    b_out = tf.Variable(self.init(shape[-1:]))
    layer = tf.add(tf.matmul(input, W_out), b_out)
    return layer


  def adapt_shapes(self, filters, pools):
    channels = self.input_shape[-1]
    result_filters, result_pools = [], []
    for filter_layer, pool in zip_longest(filters, pools):
      adapted_filters = []
      for filter in filter_layer:
        adapted_filters.append([filter[0], filter[1], channels, filter[2]])
        channels = filter[2]
      result_filters.append(adapted_filters)
      result_pools.append([1, pool[0], pool[1], 1])
    return result_filters, result_pools


  def conv_net(self):
    self.dropout_conv = tf.placeholder(tf.float32)
    self.dropout_fc = tf.placeholder(tf.float32)

    image_shape = (-1,) + self.input_shape
    layer_conv = tf.reshape(self.x, shape=image_shape)

    filters, pools = self.adapt_shapes(self.hyper_params['conv_filters'], self.hyper_params['conv_pools'])
    for filter, pool in zip(filters, pools):
      layer_conv = self.conv_layer(layer_conv, filter_size=filter, pool_size=pool, dropout=self.dropout_conv)

    layer_pool = self.reduce_layer(layer_conv)
    layer_fc = self.fully_connected_layer(layer_pool, size=self.hyper_params['fc_size'], dropout=self.dropout_fc)
    layer_out = self.output_layer(layer_fc, shape=[self.hyper_params['fc_size'], self.num_classes])

    return layer_out


  def build_graph(self, **hyper_params):
    self.hyper_params = hyper_params
    self.x = tf.placeholder(tf.float32, [None, self.input_shape[0] * self.input_shape[1] * self.input_shape[2]])
    self.y = tf.placeholder(tf.float32, [None, self.num_classes])

    prediction = self.conv_net()
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
