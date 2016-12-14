#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import copy

import operations

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
    return tf.random_normal(shape) * self.hyper_params['init_stdev']

  def _is_training(self):
    return tf.equal(self.mode, 'train')

  def _apply_activation(self, layer, name):
    func = operations.ACTIVATIONS.get(name, None)
    assert func is not None
    return func(layer)

  def _conv2d_activation(self, image, W, b, strides, params):
    layer = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, b)
    layer = operations.batch_normalization(layer, self._is_training())
    layer = self._apply_activation(layer, params.get('activation', 'relu'))
    return layer

  def _conv_layer(self, image, params):
    conv = image
    for filter in params['filters_adapted']:
      W = tf.Variable(self._init(filter))
      b = tf.Variable(self._init(filter[-1:]))
      conv = self._conv2d_activation(conv, W, b, strides=1, params=params)

    layer = tf.nn.max_pool(conv, ksize=params['pools_adapted'], strides=params['pools_adapted'], padding='SAME')
    layer = operations.dropout(layer, self._is_training(), keep_prob=params['dropout'])
    return layer

  def _reduce_layer(self, input):
    input_shape = input.get_shape()
    layer = tf.nn.avg_pool(input, ksize=[1, input_shape[1].value, input_shape[2].value, 1], strides=[1, 1, 1, 1], padding='VALID')
    return layer

  def _fully_connected_layer(self, input, size, params):
    input_shape = input.get_shape()
    fc_shape = [input_shape[1].value * input_shape[2].value * input_shape[3].value, size]
    W = tf.Variable(self._init(fc_shape))
    b = tf.Variable(self._init(fc_shape[-1:]))

    layer = tf.reshape(input, [-1, W.get_shape().as_list()[0]])
    layer = tf.add(tf.matmul(layer, W), b)
    layer = self._apply_activation(layer, params.get('activation', 'relu'))
    layer = operations.dropout(layer, self._is_training(), keep_prob=params['dropout'])
    return layer

  def _output_layer(self, input, shape):
    W_out = tf.Variable(self._init(shape))
    b_out = tf.Variable(self._init(shape[-1:]))
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

  def _build_conv_net(self):
    # Input
    image_shape = (-1,) + tuple(self.input_shape)
    layer_input = tf.reshape(self.x, shape=image_shape)

    # Conv layers
    conv_params = copy.deepcopy(self.hyper_params['conv'])  # copy because it's patched
    conv_layers_num = conv_params['layers_num']
    self._adapt_conv_shapes(conv_params, conv_layers_num)
    layer_conv = layer_input
    for i in xrange(1, conv_layers_num + 1):
      layer_conv = self._conv_layer(layer_conv, params=conv_params[i])

    # Reduced + fully-connected + output layers
    fc_params = self.hyper_params['fc']
    layer_pool = self._reduce_layer(layer_conv)
    layer_fc = self._fully_connected_layer(layer_pool, size=fc_params['size'], params=fc_params)
    layer_out = self._output_layer(layer_fc, shape=[fc_params['size'], self.num_classes])

    return layer_out

  def build_graph(self):
    self.mode = tf.placeholder(tf.string)
    self.x = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
    self.y = tf.placeholder(tf.float32, [None, self.num_classes])

    prediction = self._build_conv_net()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

    optimizer_params = self.hyper_params.get('optimizer')
    optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params.get('learning_rate', 0.001),
                                       beta1=optimizer_params.get('beta1', 0.9),
                                       beta2=optimizer_params.get('beta2', 0.999),
                                       epsilon=optimizer_params.get('epsilon', 1e-8)).minimize(cost)

    init = tf.global_variables_initializer()

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
