#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


from image_classification.tf.cifar10.main import get_cifar10_data
from image_classification.tf.conv.conv_model import ConvModel
from image_classification.tf.tensorflow_impl import TensorflowRunner
from image_classification.tf.tensorflow_impl import TensorflowSolver


hyper_params = {
  'init_sigma': 0.1,

  'optimizer': {
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 7,
    1: {
      'filters': [[3, 3, 64]],
      'activation': 'elu',
      'down_sample': {'size': [2, 2], 'pooling': 'ada_pool'},
      'residual': 1,
      'dropout': None,
    },
    2: {
      'filters': [[3, 3, 128]],
      'activation': 'elu',
      'down_sample': {'size': [2, 2], 'pooling': 'ada_pool'},
      'residual': 1,
      'dropout': 0.9,
    },
    3: {
      'filters': [[1, 1, 128],
                  [3, 3, 256]],
      'activation': 'elu',
      'down_sample': {'size': [2, 2], 'pooling': 'ada_pool'},
      'residual': 1,
      'dropout': 0.8,
    },
    4: {
      'filters': [[1, 1, 180],
                  [2, 2, 256]],
      'activation': 'elu',
      'down_sample': {'size': [2, 2], 'pooling': 'ada_pool'},
      'residual': 1,
      'dropout': 0.7,
    },
    5: {
      'filters': [[2, 2, 512]],
      'padding': 'VALID',
      'activation': 'relu',
    },
    6: {
      'filters': [[1, 1, 512]],
      'activation': 'relu',
    },
    7: {
      'filters': [[1, 1, 10]],
      'activation': 'relu',
    }
  },
}


def main():
  data = get_cifar10_data()
  model = ConvModel(input_shape=(32, 32, 3), num_classes=10, **hyper_params)
  runner = TensorflowRunner(model=model)
  solver_params = {
    'batch_size': 1000,
    'eval_batch_size': 2500,
    'epochs': 50,
    'evaluate_test': True,
    'eval_flexible': False,
  }
  solver = TensorflowSolver(data=data, runner=runner, **solver_params)
  solver.train()


if __name__ == "__main__":
  main()
