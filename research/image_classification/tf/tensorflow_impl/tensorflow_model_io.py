#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import os

import tensorflow as tf

from image_classification.tf.model_io import ModelIO


class TensorflowModelIO(ModelIO):
  def __init__(self, log_level=1, **params):
    super(TensorflowModelIO, self).__init__(log_level, **params)
    self.saver = tf.train.Saver(defer_build=True)


  def save_session(self, session, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(self.save_dir, 'session.data')
    self.saver.build()
    self.saver.save(session, destination)
    self.debug('Session saved to %s' % destination)


  def load_session(self, session, directory, log_level):
    if not directory:
      return

    directory = os.path.abspath(directory)
    session_file = os.path.join(directory, 'session.data')
    if os.path.exists(session_file):
      self.saver.build()
      self.saver.restore(session, session_file)
      self._log(log_level, 'Loaded session from %s' % session_file)
