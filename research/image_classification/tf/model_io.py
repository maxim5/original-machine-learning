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


  def save_results(self, results, directory=None):
    directory = directory or self.save_dir
    if not ModelIO._prepare(directory):
      return

    destination = os.path.join(directory, 'results.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(results))
      self.debug('Results saved to %s' % destination)


  def load_results(self, directory, log_level):
    if not directory:
      return

    results_file = os.path.join(directory, 'results.xjson')
    if os.path.exists(results_file):
      results = ModelIO._load_dict(results_file)
      self._log(log_level, 'Loaded results: %s from %s' % (dict_to_str(results), results_file))
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
    if not directory:
      return

    hyper_params = ModelIO._load_dict(os.path.join(directory, 'hyper_params.xjson'))
    if hyper_params:
      self.info('Loaded hyper-params: %s' % dict_to_str(hyper_params))
      return hyper_params


  @staticmethod
  def _prepare(directory):
    if not directory:
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
