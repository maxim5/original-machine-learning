#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf


class ConvModel:
  def __init__(self):
    self.num_input = 28 * 28
    self.num_classes = 10

    self.x = tf.placeholder(tf.float32, [None, self.num_input])
    self.y = tf.placeholder(tf.float32, [None, self.num_classes])


  def conv2d_relu(self, image, W, b, strides=1):
    image = tf.nn.conv2d(image, W, strides=[1, strides, strides, 1], padding='SAME')
    image = tf.nn.bias_add(image, b)
    return tf.nn.relu(image)


  def conv_net(self, **hyper_params):
    dropout_conv = self.dropout_conv = tf.placeholder(tf.float32)
    dropout_fc = self.dropout_fc = tf.placeholder(tf.float32)

    weights = {
      'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
      'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
      'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
      'wd1': tf.Variable(tf.random_normal([4 * 4 * 128, 1024])),
      'out': tf.Variable(tf.random_normal([1024, self.num_classes]))
    }

    biases = {
      'bc1': tf.Variable(tf.random_normal([32])),
      'bc2': tf.Variable(tf.random_normal([64])),
      'bc3': tf.Variable(tf.random_normal([128])),
      'bd1': tf.Variable(tf.random_normal([1024])),
      'out': tf.Variable(tf.random_normal([self.num_classes]))
    }

    x = tf.reshape(self.x, shape=[-1, 28, 28, 1])

    # Conv + pool + dropout
    conv1 = self.conv2d_relu(x, weights['wc1'], biases['bc1'])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout1 = tf.nn.dropout(pool1, dropout_conv)

    # Conv + pool + dropout
    conv2 = self.conv2d_relu(dropout1, weights['wc2'], biases['bc2'])
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout2 = tf.nn.dropout(pool2, dropout_conv)

    # Conv + pool + dropout
    conv3 = self.conv2d_relu(dropout2, weights['wc3'], biases['bc3'])
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout3 = tf.nn.dropout(pool3, dropout_conv)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(dropout3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_fc)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


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
