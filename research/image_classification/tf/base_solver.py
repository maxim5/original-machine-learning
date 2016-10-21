#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "maxim"


from log import Logger
from util import *


class BaseSolver(Logger):
  def __init__(self, data, runner, log_level=1, **params):
    super(BaseSolver, self).__init__(log_level)

    self.train_set = self.prepare_data(data.train)
    self.val_set = self.prepare_data(data.validation)
    self.test_set = self.prepare_data(data.test)

    self.runner = self.init_runner(runner)
    self.max_accuracy = 0

    self.epochs = params.get('epochs', 1)
    self.dynamic_epochs = params.get('dynamic_epochs')
    self.batch_size = params.get('batch_size', 16)
    self.eval_flexible = params.get('eval_flexible', True)
    self.eval_train_every = params.get('eval_train_every', 10) if not self.eval_flexible else 1e1000
    self.eval_validation_every = params.get('eval_validation_every', 100) if not self.eval_flexible else 1e1000
    self.eval_test = params.get('evaluate_test', False)
    self.eval_test_batch_size = params.get('eval_test_batch_size', self.test_set.num_examples)


  def train(self):
    with self.create_session() as session:
      self.runner.prepare(session=session)

      step = 0
      self.max_accuracy = self.init_session(session)
      while True:
        batch_x, batch_y = self.train_set.next_batch(self.batch_size)
        self.runner.run_batch(batch_x, batch_y)
        step += 1
        iteration = step * self.batch_size

        val_accuracy = self._evaluate_validation(iteration, step, batch_x, batch_y)
        if val_accuracy is not None and val_accuracy > self.max_accuracy:
          self.max_accuracy = val_accuracy
          self.on_best_accuracy(session, val_accuracy)

        if iteration >= self.train_set.num_examples * self.epochs:
          break

      self._evaluate_test()

    return self.max_accuracy


  def prepare_data(self, data_set):
    return data_set


  def init_runner(self, runner):
    return runner


  def create_session(self):
    return None


  def init_session(self, session):
    return 0


  def on_best_accuracy(self, session, accuracy):
    self._update_epochs_dynamically(accuracy)


  def _update_epochs_dynamically(self, accuracy):
    if self.dynamic_epochs is not None:
      new_epochs = self.dynamic_epochs(accuracy)
      if self.epochs != new_epochs:
        self.epochs = new_epochs or self.epochs
        self.debug('Update epochs=%d' % new_epochs)


  def _evaluate_validation(self, iteration, step, batch_x, batch_y):
    if step % self.eval_train_every == 0 and self.is_info_logged():
      loss, accuracy = self.runner.evaluate(batch_x, batch_y)
      self._log_iteration(iteration, "train_accuracy", loss, accuracy, False)

    if (step % self.eval_validation_every == 0) or (iteration % self.train_set.num_examples < self.batch_size):
      eval = self.runner.evaluate(batch_x=self.val_set.images, batch_y=self.val_set.labels)
      self._log_iteration(iteration, "validation_accuracy", eval.get('cost', 0), eval.get('accuracy', 0), True)
      return eval.get('accuracy')


  def _log_iteration(self, iteration, name, loss, accuracy, mark_best):
    marker = ' *' if mark_best and (accuracy > self.max_accuracy) else ''
    self.info("Epoch %2d, iteration %7d: loss=%.6f, %s=%.4f%s" % (self.train_set.epochs_completed,
                                                                iteration, loss, name, accuracy, marker))

  def _evaluate_test(self):
    if self.eval_test:
      for start, end in mini_batch(self.test_set.num_examples, self.eval_test_batch_size):
        eval = self.runner.evaluate(batch_x=self.test_set.images[start:end],
                                    batch_y=self.test_set.labels[start:end])
        self.info("Final test_accuracy=%.4f" % eval.get('accuracy', 0))
