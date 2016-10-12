#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def log(*msg):
  import datetime
  print '[%s]' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join([str(it) for it in msg])


# Import MNIST data
mnist = input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
train_set = mnist.train
val_set = mnist.validation
test_set = mnist.test

# Network Parameters
n_input = 784   # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
dropout_conv = tf.placeholder(tf.float32)
dropout_fc = tf.placeholder(tf.float32)


def conv2d_relu(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)


def conv_net(x, weights, biases, dropout_conv, dropout_fc):
  x = tf.reshape(x, shape=[-1, 28, 28, 1])

  # Conv + pool + dropout
  conv1 = conv2d_relu(x, weights['wc1'], biases['bc1'])
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  dropout1 = tf.nn.dropout(pool1, dropout_conv)

  # Conv + pool + dropout
  conv2 = conv2d_relu(dropout1, weights['wc2'], biases['bc2'])
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  dropout2 = tf.nn.dropout(pool2, dropout_conv)

  # Conv + pool + dropout
  conv3 = conv2d_relu(dropout2, weights['wc3'], biases['bc3'])
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


# Store layers weight & bias
weights = {
  'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
  'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
  'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
  'wd1': tf.Variable(tf.random_normal([4 * 4 * 128, 1024])),
  'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
  'bc1': tf.Variable(tf.random_normal([32])),
  'bc2': tf.Variable(tf.random_normal([64])),
  'bc3': tf.Variable(tf.random_normal([128])),
  'bd1': tf.Variable(tf.random_normal([1024])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}

# Parameters
learning_rate = 0.001
batch_size = 128

prediction = conv_net(x, weights, biases, dropout_conv, dropout_fc)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as session:
  session.run(init)
  step = 1
  while train_set.epochs_completed < 10:
    batch_x, batch_y = train_set.next_batch(batch_size)
    session.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_conv: 0.8, dropout_fc: 0.8})

    if step % 10 == 0:
      loss, acc = session.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_conv: 1.0, dropout_fc: 1.0})
      log("%d iteration %6d: loss=%10.4f, accuracy=%.4f" % (train_set.epochs_completed, step * batch_size, loss, acc))

    if step % 100 == 0:
      val_acc = session.run(accuracy, feed_dict={x: val_set.images, y: val_set.labels, dropout_conv: 1.0, dropout_fc: 1.0})
      log("val_acc=%.4f" % val_acc)
    step += 1

  test_acc = session.run(accuracy, feed_dict={x: test_set.images, y: test_set.labels, dropout_conv: 1.0, dropout_fc: 1.0})
  log("test_acc=%.4f" % test_acc)
