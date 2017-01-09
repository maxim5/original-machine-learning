#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import copy

import operations
import variable

import tensorflow as tf

# See https://github.com/tflearn/tflearn/issues/367
tf.python.control_flow_ops = tf

class ConvModel:
  def __init__(self, input_shape, num_classes, **hyper_params):
    assert hyper_params, 'Conv model hyper parameters are empty'

    self.input_shape = input_shape
    self.num_classes = num_classes
    self.hyper_params = hyper_params

  def _init(self, shape):
    return tf.random_normal(shape) * self.hyper_params['init_sigma']

  def _is_training(self):
    return tf.equal(self.mode, 'train')

  def _conv2d_activation(self, image, W, b, strides, params):
    activation = operations.ACTIVATIONS.get(params.get('activation', 'relu'))
    layer = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding=params.get('padding', 'SAME'))
    layer = tf.nn.bias_add(layer, b)
    layer = operations.batch_normalization(layer, self._is_training())
    layer = activation(layer)
    return layer

  def _conv_layer(self, input, params, index):
    conv = input
    with variable.scope('conv.%d' % index):
      for i, filter in enumerate(params['filters']):
        with variable.scope(i):
          W = variable.new(self._init(filter), name='W')
          b = variable.new(self._init(filter[-1:]), name='b')
          conv = self._conv2d_activation(conv, W, b, strides=1, params=params)

      if params.get('residual'):
        input_shape = input.get_shape().as_list()
        conv_shape = conv.get_shape().as_list()
        if input_shape[-1] != conv_shape[-1]:
          diff = conv_shape[-1] - input_shape[-1]
          pad = diff / 2
          input = tf.pad(input, [[0, 0], [0, 0], [0, 0], [pad, diff - pad]])
        conv = conv + input

      down_sample_params = params.get('down_sample')
      if down_sample_params:
        down_sample = operations.DOWN_SAMPLES.get(down_sample_params['pooling'])
        size = down_sample_params['size']
        size = [1, size[0], size[1], 1]
        layer = down_sample(conv, ksize=size, strides=size, padding='SAME')
      else:
        layer = conv

      dropout = params.get('dropout')
      if dropout:
        layer = operations.dropout(layer, self._is_training(), keep_prob=dropout)
    return layer

  def _adapt_conv_shapes(self, conv_params, conv_layers_num):
    channels = self.input_shape[-1]
    for i in xrange(1, conv_layers_num + 1):
      layer_params = conv_params[i]

      adapted = []
      for filter in layer_params['filters']:
        adapted.append([filter[0], filter[1], channels, filter[2]])
        channels = filter[2]
      layer_params['filters'] = adapted

  def _build_conv_net(self):
    image_shape = (-1,) + tuple(self.input_shape)
    layer_input = tf.reshape(self.x, shape=image_shape)

    conv_params = copy.deepcopy(self.hyper_params['conv'])  # copy because it's patched
    conv_layers_num = conv_params['layers_num']
    self._adapt_conv_shapes(conv_params, conv_layers_num)
    layer_conv = layer_input
    for i in xrange(1, conv_layers_num + 1):
      layer_conv = self._conv_layer(layer_conv, params=conv_params[i], index=i)

    layer_output = tf.reshape(layer_conv, shape=[-1, self.num_classes])
    return layer_output

  def build_graph(self):
    self.mode = tf.placeholder(tf.string, name='mode')
    self.x = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]], name='x')
    self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')

    prediction = self._build_conv_net()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

    optimizer_params = self.hyper_params.get('optimizer')
    optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params.get('learning_rate', 0.001),
                                       beta1=optimizer_params.get('beta1', 0.9),
                                       beta2=optimizer_params.get('beta2', 0.999),
                                       epsilon=optimizer_params.get('epsilon', 1e-8)).minimize(cost)

    init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    incorrect_prediction = tf.not_equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
    x_misclassified = tf.boolean_mask(self.x, incorrect_prediction)
    y_predicted = tf.boolean_mask(tf.argmax(prediction, 1), incorrect_prediction)
    y_expected = tf.boolean_mask(tf.argmax(self.y, 1), incorrect_prediction)

    return init, optimizer, cost, accuracy, x_misclassified, y_predicted, y_expected

  def params_num(self):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

  def feed_dict(self, data_set=None, x=None, y=None, mode='test'):
    if x is None and data_set is not None:
      x = data_set.x
    if y is None and data_set is not None:
      y = data_set.y
    return {
      self.x: x,
      self.y: y,
      self.mode: mode,
    }
