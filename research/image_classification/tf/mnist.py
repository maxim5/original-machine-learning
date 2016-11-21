#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime
import math
import sys

from augmentor import ImageAugmentationPlus
from conv_model import ConvModel
from data_set import Data, DataSet
from hyper_tuner import HyperTuner
from interaction import read_model
from mnist_spec import hyper_params_spec, augment_spec
from log import log
from tensorflow_impl import *
from util import random_id, dict_to_str

from image_classification.tf import spec


def get_mnist_data():
  from tensorflow.examples.tutorials import mnist
  tf_data_sets = mnist.input_data.read_data_sets("../../../dat/mnist-tf", one_hot=True)
  convert = lambda data_set: DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
  return Data(train=convert(tf_data_sets.train),
              validation=convert(tf_data_sets.validation),
              test=convert(tf_data_sets.test))


def plot_images(data, destination):
  import matplotlib.pyplot as plt

  images, labels_predicted, labels_expected = data

  num = min(len(images), 100)
  rows = int(math.sqrt(num))
  cols = (num + rows - 1) / rows

  if destination:
    plt.figure()

  f, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2), dpi=80, facecolor='w', edgecolor='k')
  axes = axes.reshape(-1)
  for i in xrange(len(axes)):
    if i < len(images):
      ax = axes[i]
      for spine in ax.spines.values():
        spine.set_edgecolor('gray')
      ax.imshow(images[i].reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest', aspect=1)
      ax.text(0, -5, 'predict %d, expect %d' % (labels_predicted[i], labels_expected[i]),
             verticalalignment='top', horizontalalignment='left', fontsize=10)
      ax.set_xticks(())
      ax.set_yticks(())
    else:
      axes[i].axis('off')

  if destination:
    destination += '.png'
    plt.savefig(destination, bbox_inches='tight')
    plt.close()
    return destination


def init_augmentation(**params):
  if params:
    log('Using augmentation params: %s' % dict_to_str(params))

    augmentation = ImageAugmentationPlus()
    scale = params.get('scale')
    if scale:
      if isinstance(scale, float):
        scale = (scale, scale)
      augmentation.add_random_scale(downscale_limit=scale[0], upscale_limit=scale[1], fix_aspect_ratio=True)

    rotation_angle = params.get('rotation_angle')
    if rotation_angle:
      augmentation.add_random_rotation(max_angle=rotation_angle)

    swirl = params.get('swirl')
    if swirl:
      augmentation.add_random_swirl(strength_limit=swirl[0], radius_limit=swirl[1])

    blur_sigma = params.get('blur_sigma')
    if blur_sigma:
      augmentation.add_random_blur(sigma_max=blur_sigma)

    crop_size = params.get('crop_size')
    if crop_size:
      augmentation.add_random_crop(crop_shape=(28, 28), padding=crop_size)
  else:
    augmentation = None
  return augmentation


def stage1():
  mnist = get_mnist_data()

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 256,
      'eval_batch_size': 5000,
      'epochs': 12,
      'dynamic_epochs': lambda acc: 5 if acc < 0.9800 else 10 if acc < 0.9920 else 15,
      'evaluate_test': True,
      'eval_flexible': False,
      'save_dir': 'model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'data_saver': plot_images,
      'save_accuracy_limit': 0.9940,
    }

    model = ConvModel(input_shape=(28, 28, 1), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    augmentation = init_augmentation(**hyper_params.get('augment', {}))
    solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, **solver_params)
    return solver

  strategy_params = {
    'io_load_dir': 'mnist/stage1-2.0',
    'io_save_dir': 'mnist/stage1-2.0',
  }

  tuner = HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
  tuner.tune()


def stage2(path=None, random_fork=True, only_test=False):
  if not path:
    path = read_model('model-zoo')

  model_path = 'model-zoo/%s' % path
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 5000,
    'epochs': 0 if only_test else 20,
    'evaluate_test': True,
    'save_dir': model_path,
    'load_dir': model_path,
    'data_saver': plot_images,
  }

  mnist = get_mnist_data()

  model_io = TensorflowModelIO(**solver_params)
  hyper_params = model_io.load_hyper_params() or {}
  if random_fork and not only_test:
    hyper_params.update({'augment': spec.get_instance(augment_spec)})

  model = ConvModel(input_shape=(28, 28, 1), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  augmentation = init_augmentation(**hyper_params.get('augment', {}))
  solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, **solver_params)
  solver.train()


def stage3(path='2016-10-24-SE5DZ8'):
  if not path:
    path = read_model('model-zoo-polish')

  model_path = 'model-zoo-polish/%s' % path
  solver_params = {
    'batch_size': 2000,
    'eval_batch_size': 5000,
    'epochs': 12,
    'evaluate_test': False,
    'eval_flexible': False,
    'save_dir': model_path,
    'load_dir': model_path,
  }

  model_io = TensorflowModelIO(**solver_params)
  hyper_params = model_io.load_hyper_params() or {}

  mnist = get_mnist_data()
  mnist.merge_validation_to_train()

  def solver_generator(augment_params):
    hyper_params['augment'] = augment_params
    model = ConvModel(input_shape=(28, 28, 1), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    augmentation = init_augmentation(**augment_params)
    solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, result_metric='avg', **solver_params)
    return solver

  strategy_params = {
    'io_load_dir': 'mnist/stage3-2.1',
    'io_save_dir': 'mnist/stage3-2.1',
  }

  tuner = HyperTuner(augment_spec, solver_generator, **strategy_params)
  tuner.tune()


def stage4(path='2016-10-24-SE5DZ8'):
  if not path:
    path = read_model('model-zoo')

  model_path = 'model-zoo-polish/%s' % path
  solver_params = {
    'batch_size': 2000,
    'eval_batch_size': 5000,
    'epochs': 80,
    'evaluate_test': False,
    'eval_flexible': False,
    'save_dir': model_path,
    'load_dir': model_path,
  }

  mnist = get_mnist_data()
  mnist.merge_validation_to_train()

  model_io = TensorflowModelIO(**solver_params)
  hyper_params = model_io.load_hyper_params() or {}

  model = ConvModel(input_shape=(28, 28, 1), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  augmentation = init_augmentation(**hyper_params.get('augment', {}))
  solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, **solver_params)
  solver.train()


if __name__ == "__main__":
  run_config = {
    'stage1': stage1,
    'stage2': stage2,
    'stage3': stage3,
    'stage4': stage4,
  }

  arguments = sys.argv
  method = run_config.get(arguments[1])
  args = arguments[2:]
  method(*args)
