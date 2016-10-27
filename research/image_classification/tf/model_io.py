#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import os
from log import Logger
from util import *


class ModelIO(Logger):
  def __init__(self, log_level=1, **params):
    super(ModelIO, self).__init__(log_level)
    self.load_dir = params.get('load_dir')
    self.save_dir = params.get('save_dir')
    self.data_saver = params.get('data_saver')


  def save_results(self, results, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'results.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(results))
      self.debug('Results saved to %s' % destination)


  def load_results(self, directory, log_level):
    if directory is None:
      return

    destination = os.path.join(directory, 'results.xjson')
    if os.path.exists(destination):
      results = ModelIO._load_dict(destination)
      self._log(log_level, 'Loaded results: %s from %s' % (dict_to_str(results), destination))
      return results


  def save_hyper_params(self, hyper_params, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'hyper_params.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(hyper_params))
      self.debug('Hyper parameters saved to %s' % destination)


  def load_hyper_params(self, directory=None):
    directory = directory or self.load_dir
    if directory is None:
      return

    hyper_params = ModelIO._load_dict(os.path.join(directory, 'hyper_params.xjson'))
    if hyper_params:
      self.info('Loaded hyper-params: %s' % dict_to_str(hyper_params))
      return hyper_params


  def save_data(self, x, y, directory=None):
    directory = directory or self.save_dir
    if self.data_saver is None or x is None or y is None or not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'misclassified')
    actual_destination = call(self.data_saver, x, y, destination)
    if actual_destination:
      self.debug('Misclassified data saved to %s' % actual_destination)
    else:
      self.warn('Data saver can not be not called or returns None')


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
