#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import datetime
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.datasets.mnist import read_data_sets

from conv_model import ConvModel
from data_set import Data, DataSet
from hyper_tuner import HyperTuner
from interaction import read_model
from tensorflow_impl import *
from util import random_id


def get_mnist_data():
  tf_data_sets = read_data_sets("../../../dat/mnist-tf", one_hot=True)
  convert = lambda data_set: DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
  return Data(train=convert(tf_data_sets.train),
              validation=convert(tf_data_sets.validation),
              test=convert(tf_data_sets.test))


def plot_images(images, labels, destination):
  num = min(len(images), 100)
  rows = int(math.sqrt(num))
  cols = (num + rows - 1) / rows

  f, axes = plt.subplots(rows, cols, figsize=(rows*2, cols*2))
  axes = axes.reshape(-1)
  for i in xrange(len(axes)):
    a = axes[i]
    a.imshow(images[i].reshape((28, 28)), cmap=plt.cm.gray_r)
    a.set_title(labels[i])
    a.set_xticks(())
    a.set_yticks(())

  if destination:
    destination += '.png'
    plt.savefig(destination, bbox_inches='tight')
    return destination


def random_conv_layer(size, num, prob=0.8):
  if np.random.uniform() > prob:
    return [[size, 1, num], [1, size, num]]
  return [[size, size, num]]


def hyper_tune_ground_up():
  activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']
  hyper_params_generator = lambda: {
    'init_stdev': np.random.uniform(0.04, 0.06),

    'augment': {
      'rotation_angle': np.random.uniform(0, 15),
      'blur_sigma': 10**np.random.uniform(-2, 0),
      'crop_size': np.random.choice(range(5)),
    },

    'optimizer': {
      'learning_rate': 10 ** np.random.uniform(-2, -4),
      'beta1': 0.9,
      'beta2': 0.999,
      'epsilon': 1e-8,
    },

    'conv': {
      'layers_num': 3,
      1: {
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([ 24,  32,  36])),
        'pools': [2, 2],
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.85, 1.0),
      },
      2: {
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([ 64,  96, 128])),
        'pools': [2, 2],
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.8, 1.0),
      },
      3: {
        'filters': random_conv_layer(size=np.random.choice([3, 5,  ]), num=np.random.choice([128, 256, 512])),
        'pools': [2, 2],
        'activation': np.random.choice(activations),
        'dropout': np.random.uniform(0.6, 1.0),
      }
    },

    'fc': {
      'size': np.random.choice([512, 768, 1024]),
      'activation': np.random.choice(activations),
      'dropout': np.random.uniform(0.5, 1.0),
    },
  }

  mnist = get_mnist_data()
  def solver_generator(hyper_params):
    solver_params = {
      'batch_size': 128,
      'eval_batch_size': 5000,
      'epochs': 12,
      'dynamic_epochs': lambda acc: 3 if acc < 0.8000 else 5 if acc < 0.9800 else 10 if acc < 0.9920 else 15,
      'evaluate_test': True,
      'save_dir': 'model-zoo/%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d'), random_id()),
      'data_saver': plot_images,
      'save_accuracy_limit': 0.9940,
    }

    augment_params = hyper_params.get('augment')
    if augment_params:
      augmentation = tflearn.ImageAugmentation()
      rotation_angle = augment_params.get('rotation_angle')
      if rotation_angle:
        augmentation.add_random_rotation(max_angle=rotation_angle)
      blur_sigma = augment_params.get('blur_sigma')
      if blur_sigma:
        augmentation.add_random_blur(sigma_max=blur_sigma)
      crop_size = augment_params.get('crop_size')
      if crop_size:
        augmentation.add_random_crop(crop_shape=(28, 28), padding=crop_size)
    else:
      augmentation = None

    model = ConvModel(input_shape=(28, 28, 1), num_classes=10, **hyper_params)
    runner = TensorflowRunner(model=model)
    solver = TensorflowSolver(data=mnist, runner=runner, augmentation=augmentation, **solver_params)
    return solver

  tuner = HyperTuner()
  tuner.tune(solver_generator, hyper_params_generator)


def fine_tune(path=None, only_test=False):
  if not path:
    path = read_model('model-zoo')

  model_path = 'model-zoo/%s' % path
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 5000,
    'epochs': 0 if only_test else 50,
    'evaluate_test': True,
    'save_dir': model_path,
    'load_dir': model_path,
    'data_saver': plot_images,
  }

  mnist = get_mnist_data()
  model = ConvModel(input_shape=(28, 28, 1), num_classes=10)
  runner = TensorflowRunner(model=model)
  solver = TensorflowSolver(data=mnist, runner=runner, **solver_params)
  solver.train()


if __name__ == "__main__":
  run_config = {
    'fine_tune': fine_tune,
    'hyper_tune_ground_up': hyper_tune_ground_up,
  }

  arguments = sys.argv
  method = run_config.get(arguments[1])
  args = arguments[2:]
  method(*args)
