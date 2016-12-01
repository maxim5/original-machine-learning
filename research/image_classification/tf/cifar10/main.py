#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

import datetime

from tflearn.datasets import cifar10

from image_classification.tf.augmentor import ImageAugmentationPlus
from image_classification.tf.conv_model import ConvModel
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

    augmentation.add_random_flip_leftright()
  else:
    augmentation = None
  return augmentation


def stage1():
  data = get_cifar10_data()

  curve_params = {
    'burn_in': 20,
    'min_input_size': 4,
    'value_limit': 0.785,
    'io_load_dir': '_models/cifar10/hyper/stage1-2.6',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.6',
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
      'save_accuracy_limit': 0.79,
    }

    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=data, runner=runner, **solver_params)
    return solver

  strategy_params = {
    'strategy': 'portfolio',
    'methods': ['ucb', 'pi', 'rand'],
    'probabilities': [0.05, 0.9, 0.05],
    'io_load_dir': '_models/cifar10/hyper/stage1-2.6',
    'io_save_dir': '_models/cifar10/hyper/stage1-2.6',
  }

  tuner = HyperTuner(hyper_params_spec_2_6, solver_generator, **strategy_params)
  tuner.tune()


def stage2(path=None, random_fork=True, batch_size=250, epochs=25):
  if not path:
    path = read_model('_models/cifar10/model-zoo-selected')

  model_path = '_models/cifar10/model-zoo-selected/%s' % path

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
    instance = spec.get_instance(augment_spec)
    hyper_params.update({'augment': instance})
    augmentation = init_augmentation(**instance)

  model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  solver = TensorflowSolver(data=data, runner=runner, augmentation=augmentation, **solver_params)
  solver.train()


# ["2016-11-29-T9FJC4"]
# "2016-11-26-IQ8F1W", "2016-11-28-IXTV2W", "2016-11-28-GF29HW", '2016-11-28-5UUDAW', '2016-11-29-7WKH7N'
def stage3(models =('2016-11-28-LUOHB2', )):
  while True:
    for path in models:
      stage2(path, batch_size=600, epochs=10)
      tf_reset_all()
      import time
      time.sleep(20)


def list_all(path='_models/cifar10/model-zoo-selected'):
  list_models(path)


if __name__ == "__main__":
  run_config = {
    'stage1': stage1,
    'stage2': stage2,
    'stage3': stage3,
    'list': list_all,
  }

  arguments = sys.argv
  method = run_config.get(arguments[1])
  args = arguments[2:]
  method(*args)
