#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

from image_classification.tf.base_io import BaseIO
from image_classification.tf.util import dict_to_str

__author__ = "maxim"


class StrategyIO(BaseIO):
  def __init__(self, strategy, **params):
    super(StrategyIO, self).__init__(**params)
    self.strategy = strategy

  def load(self):
    directory = self.load_dir
    if not directory is None:
      destination = os.path.join(directory, 'strategy-session.xjson')
      if os.path.exists(destination):
        data = StrategyIO._load_dict(destination)
        self.debug('Loaded strategy data: %s from %s' % (dict_to_str(data), destination))
        self.strategy.import_from(data)
        return

    self.strategy.import_from({})

  def save(self):
    directory = self.save_dir
    if not StrategyIO._prepare(directory):
      return

    destination = os.path.join(directory, 'strategy-session.xjson')
    with open(destination, 'w') as file_:
      file_.write(dict_to_str(self.strategy.export_to()))
      self.debug('Strategy data saved to %s' % destination)
