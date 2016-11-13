#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from log import Logger
from util import str_to_dict

__author__ = "maxim"


class BaseIO(Logger):
  def __init__(self, log_level=1, **params):
    super(BaseIO, self).__init__(log_level)
    self.load_dir = params.get('load_dir')
    self.save_dir = params.get('save_dir')

  @staticmethod
  def _prepare(directory):
    if directory is None:
      return False
    if not os.path.exists(directory):
      os.makedirs(directory)
    return True

  @staticmethod
  def _load_dict(from_file):
    if not os.path.exists(from_file):
      return {}
    try:
      with open(from_file, 'r') as file_:
        line = file_.readline()
        return str_to_dict(line)
    except:
      return {}
