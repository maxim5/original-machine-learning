#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import copy

import tensorflow as tf
import tflearn

# See https://github.com/tflearn/tflearn/issues/367
tf.python.control_flow_ops = tf


class ConvModel:
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes


  def init(self, shape):
    return tf.random_normal(shape) * self.hyper_params['init_stdev']


  def train_or_test(self, train_func, test_func):
    return tf.cond(tf.equal(self.mode, 'train'), train_func, test_func)


  def _get_activation_function(self, name):
    return getattr(tflearn.activations, name, None)


  def conv2d_activation(self, image, W, b, strides, params, cache):
    activation_func = self._get_activation_function(params.get('activation', 'relu'))
    layer = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, b)

    bn_input = layer
    def train_batchnorm():
      normalized = tflearn.batch_normalization(bn_input)
      cache['bn'] = (normalized.beta, normalized.gamma)
      return normalized
    def test_batchnorm():
      beta, gamma = cache['bn']
      normalized = bn_input * gamma + beta
      return normalized
    layer = self.train_or_test(train_batchnorm, test_batchnorm)

    layer = activation_func(layer)
    return layer


  def conv_layer(self, image, params, cache):
    conv = image
    for filter in params['filters_adapted']:
      W = tf.Variable(self.init(filter))
      b = tf.Variable(self.init(filter[-1:]))
      conv = self.conv2d_activation(conv, W, b, strides=1, params=params, cache=cache)

    layer = tf.nn.max_pool(conv, ksize=params['pools_adapted'], strides=params['pools_adapted'], padding='SAME')
    layer = self.train_or_test(lambda: tf.nn.dropout(layer, keep_prob=params['dropout']), lambda: layer)
    return layer


  def reduce_layer(self, input):
    input_shape = input.get_shape()
    layer = tf.nn.avg_pool(input, ksize=[1, input_shape[1].value, input_shape[2].value, 1], strides=[1, 1, 1, 1], padding='VALID')
    return layer


  def fully_connected_layer(self, input, size, params):
    input_shape = input.get_shape()
    fc_shape = [input_shape[1].value * input_shape[2].value * input_shape[3].value, size]
    W = tf.Variable(self.init(fc_shape))
    b = tf.Variable(self.init(fc_shape[-1:]))

    activation_func = self._get_activation_function(params.get('activation', 'relu'))
    layer = tf.reshape(input, [-1, W.get_shape().as_list()[0]])
    layer = tf.add(tf.matmul(layer, W), b)
    layer = activation_func(layer)
    layer = self.train_or_test(lambda: tf.nn.dropout(layer, keep_prob=params['dropout']), lambda: layer)
    return layer


  def output_layer(self, input, shape):
    W_out = tf.Variable(self.init(shape))
    b_out = tf.Variable(self.init(shape[-1:]))
    layer = tf.add(tf.matmul(input, W_out), b_out)
    return layer


  def _adapt_conv_shapes(self, conv_params, conv_layers_num):
    channels = self.input_shape[-1]
    for i in xrange(1, conv_layers_num + 1):
      layer_params = conv_params[i]

      adapted = []
      for filter in layer_params['filters']:
        adapted.append([filter[0], filter[1], channels, filter[2]])
        channels = filter[2]
      layer_params['filters_adapted'] = adapted

      pools = layer_params['pools']
      layer_params['pools_adapted'] = [1, pools[0], pools[1], 1]


  def conv_net(self):
    # Input
    image_shape = (-1,) + tuple(self.input_shape)
    layer_input = tf.reshape(self.x, shape=image_shape)

    # Conv layers
    conv_params = self.hyper_params['conv']
    conv_layers_num = conv_params['layers_num']
    self._adapt_conv_shapes(conv_params, conv_layers_num)
    layer_conv = layer_input
    for i in xrange(1, conv_layers_num + 1):
      self.cache.setdefault(i, {})
      layer_conv = self.conv_layer(layer_conv, params=conv_params[i], cache=self.cache[i])

    # Reduced + fully-connected + output layers
    fc_params = self.hyper_params['fc']
    layer_pool = self.reduce_layer(layer_conv)
    layer_fc = self.fully_connected_layer(layer_pool, size=fc_params['size'], params=fc_params)
    layer_out = self.output_layer(layer_fc, shape=[fc_params['size'], self.num_classes])

    return layer_out


  def build_graph(self, **hyper_params):
    self.mode = tf.placeholder(tf.string)
    self.hyper_params = copy.deepcopy(hyper_params)
    self.x = tf.placeholder(tf.float32, [None, self.input_shape[0] * self.input_shape[1] * self.input_shape[2]])
    self.y = tf.placeholder(tf.float32, [None, self.num_classes])
    self.cache = {}

    prediction = self.conv_net()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

    adam_params = self.hyper_params['adam']
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_params.get('learning_rate', 0.001),
                                       beta1=adam_params.get('beta1', 0.9),
                                       beta2=adam_params.get('beta2', 0.999),
                                       epsilon=adam_params.get('epsilon', 1e-8)).minimize(cost)

    init = tf.initialize_all_variables()

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    incorrect_prediction = tf.not_equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
    misclassified_x = tf.boolean_mask(self.x, incorrect_prediction)
    misclassified_y = tf.boolean_mask(tf.argmax(prediction, 1), incorrect_prediction)

    return init, optimizer, cost, accuracy, misclassified_x, misclassified_y


  def params_num(self):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


  def feed_dict(self, data_set=None, images=None, labels=None, mode='test'):
    if images is None and data_set is not None:
      images = data_set.images
    if labels is None and data_set is not None:
      labels = data_set.labels
    return {
      self.x: images,
      self.y: labels,
      self.mode: mode,
    }
