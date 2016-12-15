#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import tensorflow as tf

scope_stack = []

class scope:
  def __init__(self, s):
    self._scope = s
    self._stack = scope_stack

  def __enter__(self):
    self._stack.append(str(self._scope))

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._stack.pop()

def get_full_name(name):
  return name if not scope_stack or name is None else '.'.join(scope_stack) + '/' + name

def new(value, shape=None, name=None, trainable=True):
  if shape is not None:
    if value == 0:
      initial_value = tf.zeros(shape=shape)
    else:
      initial_value = tf.ones(shape=shape) * value
  else:
    initial_value = value
  return tf.Variable(initial_value=initial_value, trainable=trainable, name=get_full_name(name))
