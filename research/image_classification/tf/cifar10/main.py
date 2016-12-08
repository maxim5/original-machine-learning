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


# {'conv': {1: {'activation': 'elu', 'dropout': 0.850045, 'filters': [[5, 1, 64], [1, 5, 64]], 'pools': [2, 2]}, 2: {'activation': 'relu', 'dropout': 0.907284, 'filters': [[5, 1, 105], [1, 7, 101]], 'pools': [2, 2]}, 3: {'activation': 'elu', 'dropout': 0.889042, 'filters': [[7, 1, 190], [1, 7, 214]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'prelu', 'dropout': 0.699671, 'size': 439}, 'init_stdev': 0.075323, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.001575}}
# {'conv': {1: {'activation': 'elu', 'dropout': 0.850263, 'filters': [[3, 1, 49], [1, 7, 60]], 'pools': [2, 2]}, 2: {'activation': 'relu', 'dropout': 0.856936, 'filters': [[5, 1, 109], [1, 7, 117]], 'pools': [2, 2]}, 3: {'activation': 'relu6', 'dropout': 0.816066, 'filters': [[7, 1, 223], [1, 5, 195]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'elu', 'dropout': 0.956791, 'size': 489}, 'init_stdev': 0.075416, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.001385}}
# {'conv': {1: {'activation': 'elu', 'dropout': 0.914531, 'filters': [[3, 1, 65], [1, 3, 63]], 'pools': [2, 2]}, 2: {'activation': 'leaky_relu', 'dropout': 0.874635, 'filters': [[3, 1, 158], [1, 3, 141]], 'pools': [2, 2]}, 3: {'activation': 'prelu', 'dropout': 0.879949, 'filters': [[3, 1, 188], [1, 3, 215]], 'pools': [2, 2]}, 'layers_num': 3}, 'fc': {'activation': 'relu6', 'dropout': 0.522513, 'size': 510}, 'init_stdev': 0.093356, 'optimizer': {'beta1': 0.900000, 'beta2': 0.999000, 'epsilon': 1.000000e-08, 'learning_rate': 0.001445}}
def stage1():
  data = get_cifar10_data()
  directory = '_models/cifar10/hyper/stage1-4.0'

  curve_params = {
    'burn_in': 20,
    'min_input_size': 4,
    'value_limit': 0.785,
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
      'save_accuracy_limit': 0.795,
    }

    model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=data, runner=runner, **solver_params)
    return solver

  strategy_params = {
    'strategy': 'portfolio',
    'methods': ['ucb', 'pi', 'rand'],
    'probabilities': [0.1, 0.9, 0.0],
    'io_load_dir': directory,
    'io_save_dir': directory,
  }

  tuner = HyperTuner(hyper_params_spec_4_0, solver_generator, **strategy_params)
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


# ["2016-11-29-T9FJC4"]
# "2016-11-26-IQ8F1W", "2016-11-28-IXTV2W", "2016-11-28-GF29HW", '2016-11-28-5UUDAW', '2016-11-29-7WKH7N'
def stage3(models=('2016-12-06-1ZU4GB', '2016-12-06-E7S0E1', '2016-12-06-XSKFM8', '2016-12-07-7OG78H', '2016-12-07-V0EEVZ')):
  while True:
    for path in models:
      stage2(path, batch_size=500, epochs=10)
      tf_reset_all()
      import time
      time.sleep(20)


def stage4(path='_models/cifar10/model-zoo/2016-12-07-7OG78H'):
  solver_params = {
    'batch_size': 500,
    'eval_batch_size': 2500,
    'epochs': 5,
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

  model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  augmentation = init_augmentation(**hyper_params.get('augment', {}))
  solver = TensorflowSolver(data=data, runner=runner, augmentation=augmentation, **solver_params)
  solver.train()


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
