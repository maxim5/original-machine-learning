#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from image_classification.tf.base_runner import BaseRunner
from image_classification.tf.util import *


class TensorflowRunner(BaseRunner):
  def __init__(self, model, log_level=1):
    super(TensorflowRunner, self).__init__(log_level)
    self.model = model
    self.session = None


  def init_model(self, **hyper_params):
    self.model.hyper_params = hyper_params


  def build_model(self, **kwargs):
    self.session = kwargs['session']
    init, self.optimizer, self.cost, self.accuracy, self.x_misclassified, self.y_predicted, self.y_expected = \
      self.model.build_graph()
    self.info('Start training. Model size: %dk' % (self.model.params_num() / 1000))
    self.info('Hyper params: %s' % dict_to_str(self.model.hyper_params))
    self.session.run(init)


  def run_batch(self, batch_x, batch_y):
    self.session.run(self.optimizer, feed_dict=self.model.feed_dict(x=batch_x, y=batch_y, mode='train'))


  def evaluate(self, batch_x, batch_y):
    cost, accuracy, x, y_predicted, y_expected = \
      self.session.run([self.cost, self.accuracy, self.x_misclassified, self.y_predicted, self.y_expected],
                       feed_dict=self.model.feed_dict(x=batch_x, y=batch_y, mode='test'))
    return {'cost': cost, 'accuracy': accuracy, 'data': (x, y_predicted, y_expected)}


  def describe(self):
    return {'model_size': self.model.params_num(), 'hyper_params': self.model.hyper_params}
