#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "maxim"


from log import Logger
from util import *


class BaseSolver(Logger):
  def __init__(self, data, runner, augmentation=None, log_level=1, **params):
    super(BaseSolver, self).__init__(log_level)

    data.reset_counters()
    self.train_set = data.train
    self.val_set = data.validation
    self.test_set = data.test
    self.augmentation = augmentation

    self.runner = self.init_runner(runner)
    self.max_val_accuracy = 0

    self.epochs = params.get('epochs', 1)
    self.dynamic_epochs = params.get('dynamic_epochs')
    self.batch_size = params.get('batch_size', 16)
    self.eval_batch_size = params.get('eval_batch_size', self.val_set.size)
    self.eval_flexible = params.get('eval_flexible', True)
    self.eval_train_every = params.get('eval_train_every', 10) if not self.eval_flexible else 1e1000
    self.eval_validation_every = params.get('eval_validation_every', 100) if not self.eval_flexible else 1e1000
    self.eval_test = params.get('evaluate_test', False)


  def train(self):
    with self.create_session() as session:
      self.runner.build_model(session=session)

      self.max_val_accuracy = self.init_session()
      while self.train_set.epochs_completed < self.epochs:
        batch_x, batch_y = self.train_set.next_batch(self.batch_size)
        batch_x = self.augment(batch_x)
        self.runner.run_batch(batch_x, batch_y)

        eval_result = self._evaluate_validation(batch_x, batch_y)
        val_accuracy = eval_result.get('accuracy') if eval_result is not None else None
        if val_accuracy is not None and val_accuracy > self.max_val_accuracy:
          self.max_val_accuracy = val_accuracy
          self.on_best_accuracy(val_accuracy, eval_result)

      if self.eval_test:
        self._evaluate_test()

    return self.max_val_accuracy


  def init_runner(self, runner):
    return runner


  def create_session(self):
    return None


  def init_session(self):
    return 0


  def augment(self, x):
    augmented = call(self.augmentation, x)
    return augmented if augmented is not None else x


  def on_best_accuracy(self, accuracy, eval_result):
    self._update_epochs_dynamically(accuracy)


  def _update_epochs_dynamically(self, accuracy):
    if self.dynamic_epochs is not None:
      new_epochs = self.dynamic_epochs(accuracy)
      if self.epochs != new_epochs:
        self.epochs = new_epochs or self.epochs
        self.debug('Update epochs=%d' % new_epochs)


  def _evaluate_validation(self, batch_x, batch_y):
    if (self.train_set.step % self.eval_train_every == 0) and self.is_info_logged():
      eval_ = self.runner.evaluate(batch_x, batch_y)
      self._log_iteration('train_accuracy', eval_.get('cost', 0), eval_.get('accuracy', 0), False)

    if (self.train_set.step % self.eval_validation_every == 0) or self.train_set.just_completed:
      eval_ = self._evaluate(batch_x=self.val_set.x, batch_y=self.val_set.y)
      self._log_iteration('validation_accuracy', eval_.get('cost', 0), eval_.get('accuracy', 0), True)
      return eval_


  def _evaluate_test(self):
    eval_ = self._evaluate(batch_x=self.test_set.x, batch_y=self.test_set.y)
    self.info('Final test_accuracy=%.4f' % eval_.get('accuracy', 0))
    return eval_


  def _log_iteration(self, name, loss, accuracy, mark_best):
    marker = ' *' if mark_best and (accuracy > self.max_val_accuracy) else ''
    self.info('Epoch %2d, iteration %7d: loss=%.6f, %s=%.4f%s' %
              (self.train_set.epochs_completed, self.train_set.index, loss, name, accuracy, marker))


  def _evaluate(self, batch_x, batch_y):
    size = len(batch_x)
    assert len(batch_y) == size

    if size <= self.eval_batch_size:
      return self.runner.evaluate(batch_x, batch_y)

    result = {'accuracy': 0, 'cost': 0, 'misclassified_x': [], 'misclassified_y': []}
    for start, end in mini_batch(size, self.eval_batch_size):
      eval = self.runner.evaluate(batch_x=batch_x[start:end], batch_y=batch_y[start:end])
      result['accuracy'] += eval.get('accuracy', 0) * len(batch_x[start:end])
      result['cost'] += eval.get('cost', 0) * len(batch_x[start:end])
      result['misclassified_x'].append(eval.get('misclassified_x'))
      result['misclassified_y'].append(eval.get('misclassified_y'))
    result['accuracy'] /= size
    result['cost'] /= size
    result['misclassified_x'] = safe_concat(result['misclassified_x'])
    result['misclassified_y'] = safe_concat(result['misclassified_y'])

    return result
