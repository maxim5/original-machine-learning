#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import tensorflow as tf


def log(*msg):
  import datetime
  print '[%s]' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join([str(it) for it in msg])


def is_gpu():
  from tensorflow.python.client import device_lib
  local_devices = device_lib.list_local_devices()
  return len([x for x in local_devices if x.device_type == 'GPU']) > 0


def total_params():
  total_parameters = 0
  for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      variable_parametes = 1
      for dim in shape:
          variable_parametes *= dim.value
      total_parameters += variable_parametes
  return total_parameters


def train(data_sets, model, **hyper_params):
  train_set = data_sets.train
  val_set = data_sets.validation
  test_set = data_sets.test

  optimizer, cost, accuracy, init = model.build_graph(**hyper_params)
  log("Total parameters: %dk" % (total_params() / 1000))
  log("Hyper params: ", hyper_params)

  with tf.Session() as session:
    log("Start training")
    session.run(init)

    step = 1
    is_gpu_used = is_gpu()
    epochs = hyper_params['epochs']
    batch_size = hyper_params['batch_size']
    while train_set.epochs_completed < epochs:
      batch_x, batch_y = train_set.next_batch(batch_size)
      session.run(optimizer, feed_dict=model.feed_dict(images=batch_x, labels=batch_y, **hyper_params))

      loss, acc, name = None, None, None
      if is_gpu_used:
        if step % 500 == 0:
          loss, acc = session.run([cost, accuracy], feed_dict=model.feed_dict(data_set=val_set))
          name = "validation_accuracy"
      elif step % 100 == 0:
        loss, acc = session.run([cost, accuracy], feed_dict=model.feed_dict(data_set=val_set))
        name = "validation_accuracy"
      elif step % 10 == 0:
        loss, acc = session.run([cost, accuracy], feed_dict=model.feed_dict(images=batch_x, labels=batch_y))
        name = "train_accuracy"
      if loss is not None and acc is not None and name is not None:
        log("epoch %d, iteration %6d: loss=%10.3f, %s=%.4f" % (train_set.epochs_completed, step * batch_size, loss, name, acc))
      step += 1

    test_acc = session.run(accuracy, feed_dict=model.feed_dict(data_set=test_set))
    log("Final test_accuracy=%.4f" % test_acc)

    return test_acc
