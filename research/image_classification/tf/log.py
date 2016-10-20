#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime


def log(*msg):
  print '[%s]' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join([str(it) for it in msg])


class Logger(object):
  def __init__(self, log_level=1):
    assert type(log_level) == int
    self.log_level = log_level

  def silence(self):
    self.log_level = 10

  def verbose(self, level=1):
    self.log_level = -level

  def debug(self, *msg):
    self._log(0, *msg)

  def info(self, *msg):
    self._log(1, *msg)

  def warn(self, *msg):
    self._log(2, *msg)

  def vlog(self, *msg):
    self._log(-1, *msg)

  def vlog2(self, *msg):
    self._log(-2, *msg)

  def vlog3(self, *msg):
    self._log(-3, *msg)

  def _log(self, level, *msg):
    if level >= self.log_level:
      log(*msg)

  def is_info_logged(self):
    return self.log_level >= 1
