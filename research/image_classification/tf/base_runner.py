#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from log import Logger


class BaseRunner(Logger):
  def __init__(self, log_level=1):
    super(BaseRunner, self).__init__(log_level)


  def prepare(self, **kwargs):
    pass


  def run_batch(self, batch_x, batch_y):
    pass


  def evaluate(self, batch_x, batch_y):
    return {}
