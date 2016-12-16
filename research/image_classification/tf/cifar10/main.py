#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import datetime

from tflearn.datasets import cifar10

from image_classification.tf.augmentor import ImageAugmentationPlus
from image_classification.tf.conv.conv_model import ConvModel
from image_classification.tf.hyper_tuner import HyperTuner, tf_reset_all
from image_classification.tf.tensorflow_impl import TensorflowRunner, TensorflowSolver
from image_classification.tf.data_set import Data, DataSet
from image_classification.tf.util import *
from image_classification.tf.curve_predictor import LinearCurvePredictor
from image_classification.tf.interaction import *
from image_classification.tf.tensorflow_impl import TensorflowModelIO

from cifar_spec import *


def get_cifar10_data(validation_size=5000, one_hot=True):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data('../../../dat/cifar-10-tf', one_hot=one_hot)

  x_val = x_train[:validation_size]
  y_val = y_train[:validation_size]
  x_train = x_train[validation_size:]
  y_train = y_train[validation_size:]

  return Data(train=DataSet(x_train, y_train),
              validation=DataSet(x_val, y_val),
              test=DataSet(x_test, y_test))


def init_augmentation(**params):
  if params:
    log('Using augmentation params: %s' % dict_to_str(params))

    augmentation = ImageAugmentationPlus()
    scale = params.get('scale')
    if scale:
      if isinstance(scale, float):
        scale = (scale, scale)
      augmentation.add_random_scale(downscale_limit=scale[0], upscale_limit=scale[1], fix_aspect_ratio=True)

    crop_size = params.get('crop_size')
    if crop_size:
      augmentation.add_random_crop(crop_shape=(32, 32), padding=crop_size)

    brightness = params.get('brightness')
    if brightness:
      augmentation.add_random_brightness(brightness[0], brightness[1])

    contrast = params.get('contrast')
    if contrast:
      augmentation.add_random_contrast(contrast[0], contrast[1])

    augmentation.add_random_flip_leftright()
  else:
    augmentation = None
  return augmentation


def stage1():
  data = get_cifar10_data()
  directory = '_models/cifar10/hyper/stage1-5.0'

  curve_params = {
    'burn_in': 20,
    'min_input_size': 4,
    'value_limit': 0.75,
    'io_load_dir': directory,
    'io_save_dir': directory,
  }
  curve_predictor = LinearCurvePredictor(**curve_params)

  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 250,
      'eval_batch_size': 2500,
      'epochs': 15,
      'stop_condition': curve_predictor.stop_condition(),
      'result_metric': curve_predictor.result_metric(),
      'evaluate_test': True,
      'eval_flexible': False,
      'save_dir': '_models/cifar10/model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'save_accuracy_limit': 0.8,
    }

    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=data, runner=runner, **solver_params)
    return solver

  strategy_params = {
    'strategy': 'portfolio',
    'methods': ['ucb', 'pi', 'rand'],
    'probabilities': [0.5, 0.0, 0.5],
    'io_load_dir': directory,
    'io_save_dir': directory,
  }

  tuner = HyperTuner(hyper_params_spec_5_0, solver_generator, **strategy_params)
  tuner.tune()


def stage2(path=None, random_fork=True, batch_size=250, epochs=25):
  if not path:
    path = read_model('_models/cifar10/model-zoo')

  model_path = '_models/cifar10/model-zoo/%s' % path

  data = get_cifar10_data()

  solver_params = {
    'batch_size': batch_size,
    'eval_batch_size': 2500,
    'epochs': epochs,
    'evaluate_test': True,
    'eval_flexible': True,
    'save_dir': model_path,
    'load_dir': model_path,
  }

  model_io = TensorflowModelIO(**solver_params)
  hyper_params = model_io.load_hyper_params() or {}

  augmentation = None
  if random_fork:
    # instance = spec.get_instance(augment_spec)
    instance = {'crop_size': 2, 'scale': [0.9, 1.2]}
    hyper_params.update({'augment': instance})
    augmentation = init_augmentation(**instance)

  model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  solver = TensorflowSolver(data=data, runner=runner, augmentation=augmentation, **solver_params)
  solver.train()


def stage3(models=('',)):
  while True:
    for path in models:
      tf_reset_all()
      stage2(path, batch_size=500, epochs=10)
      import time
      time.sleep(20)


def stage4(path='_models/cifar10/model-zoo/2016-12-07-7OG78H'):
  solver_params = {
    'batch_size': 500,
    'eval_batch_size': 2500,
    'epochs': 3,
    'evaluate_test': False,
    'eval_flexible': False,
    'eval_train_every': 2,
    'eval_validation_every': 2,
    'save_dir': path,
    'load_dir': path,
  }

  data = get_cifar10_data()
  data.merge_validation_to_train()

  model_io = TensorflowModelIO(**solver_params)
  hyper_params = model_io.load_hyper_params() or {}

  def solver_generator(augment_params):
    hyper_params['augment'] = augment_params
    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    augmentation = init_augmentation(**hyper_params['augment'])
    solver = TensorflowSolver(data=data, runner=runner, augmentation=augmentation, result_metric='avg', **solver_params)
    return solver

  strategy_params = {
    'io_load_dir': '_models/cifar10/hyper/stage2-1.0',
    'io_save_dir': '_models/cifar10/hyper/stage2-1.0',
  }

  tuner = HyperTuner(augment_spec, solver_generator, **strategy_params)
  tuner.tune()


def list_all(path='_models/cifar10/model-zoo'):
  list_models(path)


if __name__ == "__main__":
  run_config = {
    'stage1': stage1,
    'stage2': stage2,
    'stage3': stage3,
    'stage4': stage4,
    'list': list_all,
  }

  arguments = sys.argv
  method = run_config.get(arguments[1])
  args = arguments[2:]
  method(*args)
