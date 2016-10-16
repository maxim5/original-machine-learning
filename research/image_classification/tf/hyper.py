#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import copy

from log import *
from util import *

import tensorflow as tf
from tensorflow.python.client import device_lib


def tf_reset_data_set(data_set):
  data_set._epochs_completed = 0
  data_set._index_in_epoch = 0
  return data_set


def tf_is_gpu():
  local_devices = device_lib.list_local_devices()
  return len([x for x in local_devices if x.device_type == 'GPU']) > 0

is_gpu_available = tf_is_gpu()


class Solver(Logger):
  def __init__(self, data, model, log_level=1):
    super(Solver, self).__init__(log_level)
    self.data = data
    self.model = model


  def train(self, **hyper_params):
    epochs = hyper_params['epochs']
    batch_size = hyper_params['batch_size']
    log_accuracy_flexible = hyper_params.get('log_accuracy_flexible', True)
    log_train_every = hyper_params.get('log_train_every', 10)
    log_validation_every = hyper_params.get('log_validation_every', 100)
    eval_test = hyper_params.get('evaluate_test', False)

    log_every = not log_accuracy_flexible or not is_gpu_available

    train_set = tf_reset_data_set(self.data.train)
    val_set = self.data.validation
    test_set = self.data.test

    model = self.model
    optimizer, cost, accuracy, init = model.build_graph(**hyper_params)

    with tf.Session() as session:
      self.info("Start training. Model size: %dk" % (model.params_num() / 1000))
      self.debug("Hyper params: %s" % dict_to_str(hyper_params))
      session.run(init)

      step = 0
      max_val_acc = 0
      while True:
        batch_x, batch_y = train_set.next_batch(batch_size)
        session.run(optimizer, feed_dict=model.feed_dict(images=batch_x, labels=batch_y, mode='train'))
        step += 1
        iteration = step * batch_size

        compose_msg = lambda iteration_, name_, loss_, accuracy_: \
                "epoch %2d, iteration %7d: loss=%.6f, %s=%.4f" % \
                (train_set.epochs_completed, iteration_, loss_, name_, accuracy_)

        if log_every and step % log_train_every == 0 and self.is_info_logged():
          loss, train_accuracy = session.run([cost, accuracy], feed_dict=model.feed_dict(images=batch_x, labels=batch_y))
          self.info(compose_msg(iteration, "train_accuracy", loss, train_accuracy))

        if (log_every and step % log_validation_every == 0) or \
           (not log_every and iteration % train_set.num_examples < batch_size):
          loss, train_accuracy = session.run([cost, accuracy], feed_dict=model.feed_dict(data_set=val_set))
          max_val_acc = max(max_val_acc, train_accuracy)
          self.info(compose_msg(iteration, "validation_accuracy", loss, train_accuracy))

        if iteration >= train_set.num_examples * epochs:
          break

      if eval_test:
        test_accuracy = session.run(accuracy, feed_dict=model.feed_dict(data_set=test_set))
        log("Final test_accuracy=%.4f" % test_accuracy)
        return max_val_acc, test_accuracy

    return max_val_acc


def tf_reset_all():
  tf.reset_default_graph()


class HyperTuner(Logger):
  def __init__(self, solver, save_path=None, save_limit=0.993, log_level=1):
    super(HyperTuner, self).__init__(log_level)
    self.solver = solver
    self.best_accuracy = 0

    self.save_path = save_path % save_limit if save_path else None
    self.save_limit = save_limit


  def _save_to_disk(self, accuracy, all_hyper_params):
    if not self.save_path or accuracy < self.save_limit:
      return

    self.debug('Saving hyper_params to %s' % self.save_path)
    with open(self.save_path, 'a') as file_:
      file_.writelines([
        '# max_validation_accuracy=%.4f\n' % accuracy,
        '# model_complexity=%d\n' % self.solver.model.params_num(),
        '%s\n' % dict_to_str(all_hyper_params),
        '\n',
      ])


  def _update_accuracy(self, trial, accuracy, tuned_params):
    marker = ' '
    if accuracy > self.best_accuracy:
      self.best_accuracy = accuracy
      marker = '!'
    self.info('[%d] %s accuracy=%.4f, params: %s' % (trial, marker, accuracy, dict_to_str(tuned_params)))


  def tune(self, fixed_params, tuned_params_generator):
    self.info('Start hyper-tuner')

    trial = 0
    while True:
      hyper_params = copy.deepcopy(fixed_params)
      tuned_params = tuned_params_generator()
      deep_update(hyper_params, tuned_params)

      trial += 1
      tf_reset_all()
      accuracy = self.solver.train(**hyper_params)
      self._update_accuracy(trial, accuracy, tuned_params)

      self._save_to_disk(accuracy, hyper_params)


class HyperParamsFile(Logger):
  def __init__(self, path, log_level=1):
    super(HyperParamsFile, self).__init__(log_level)
    self.path = path
    self.packs = None
    self.hyper_list = []


  def _load_all(self):
    if self.packs is None:
      with open(self.path, 'r') as file_:
        raw_lines = file_.readlines()

      self.packs = []
      pack = []
      for line in raw_lines:
        line = line.strip()
        if len(line) == 0:
          if len(pack) > 0:
            self.packs.append(pack)
            pack = []
        else:
          pack.append(line)
          if line.startswith('#'):
            pass
          else:
            self.hyper_list.append(str_to_dict(line))

      if len(pack) > 0:
        self.packs.append(pack)


  def get_all(self):
    self._load_all()
    return self.hyper_list


  def update_pack(self, index, line):
    assert len(line) > 0
    if not line.startswith('#'):
      line = '# %s' % line
    self.packs[index].insert(-1, line)


  def save_all(self):
    with open(self.path, 'w') as file_:
      for pack in self.packs:
        for line in pack:
          file_.write(line)
          file_.write('\n')
        file_.write('\n')