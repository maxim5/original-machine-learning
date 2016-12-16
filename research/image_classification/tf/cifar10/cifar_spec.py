#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


from image_classification.tf import spec


def random_conv_layer(size, num):
  return [[size, size, num]]

def random_deep_conv(size, num):
  return [[size, 1, num], [1, size, num]]

def random_conv_layer_plus(size, num, prob=0.8):
  def if_cond(switch, size, num):
    if switch > prob:
      return [[size, 1, num], [1, size, num]]
    return [[size, size, num]]

  return spec.merge([spec.uniform(), size, num], if_cond)

activations = ['relu', 'relu6', 'elu', 'prelu', 'leaky_relu']


hyper_params_spec_2_0 = {
  'init_stdev': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -3),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_conv_layer_plus(size=spec.choice(range(3, 8)), num=spec.choice(range(24, 41))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.9, 1.0),
    },
    2: {
      'filters': random_conv_layer_plus(size=spec.choice(range(3, 8)), num=spec.choice(range(64, 121))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.8, 1.0),
    },
    3: {
      'filters': random_conv_layer(size=3, num=spec.choice(range(128, 257))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(320, 765)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.5, 1.0),
  },
}


hyper_params_spec_2_5 = {
  'init_stdev': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_deep_conv(size=spec.choice(range(3, 10)), num=spec.choice(range(40, 65))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': random_deep_conv(size=spec.choice(range(3, 10)), num=spec.choice(range(100, 161))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.8, 1.0),
    },
    3: {
      'filters': random_deep_conv(size=spec.choice(range(3, 6)),  num=spec.choice(range(200, 401))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(400, 765)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.5, 1.0),
  },
}


hyper_params_spec_2_6 = {
  'init_stdev': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': random_deep_conv(size=spec.choice(range(3, 10)), num=spec.choice(range(40, 65))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': random_deep_conv(size=spec.choice(range(3, 10)), num=spec.choice(range(100, 161))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.75, 1.0),
    },
    3: {
      'filters': random_deep_conv(size=spec.choice(range(3, 6)),  num=spec.choice(range(200, 401))),
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(400, 765)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.5, 1.0),
  },
}


hyper_params_spec_3_0 = {
  'init_stdev': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': [[spec.choice([3, 5, 7]), 1, spec.choice(range(40, 65))],
                  [1, spec.choice([3, 5, 7]), spec.choice(range(40, 65))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': [[spec.choice([3, 5, 7]), 1, spec.choice(range(80, 125))],
                  [1, spec.choice([3, 5, 7]), spec.choice(range(80, 125))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.75, 1.0),
    },
    3: {
      'filters': [[spec.choice([3, 5, 7]), 1, spec.choice(range(160, 257))],
                  [1, spec.choice([3, 5, 7]), spec.choice(range(160, 257))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(400, 641)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.45, 1.0),
  },
}


hyper_params_spec_4_0 = {
  'init_stdev': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': [[3, 1, spec.choice(range(60, 70))],
                  [1, 3, spec.choice(range(60, 70))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': [[3, 1, spec.choice(range(120, 180))],
                  [1, 3, spec.choice(range(120, 180))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.75, 1.0),
    },
    3: {
      'filters': [[3, 1, spec.choice(range(180, 257))],
                  [1, 3, spec.choice(range(180, 257))]],
      'pools': [2, 2],
      'activation': spec.choice(activations),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'fc': {
    'size': spec.choice(range(400, 641)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.45, 1.0),
  },
}


poolings = ['max_pool', 'avg_pool', 'ada_pool']

hyper_params_spec_5_0 = {
  'init_sigma': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
  },

  'conv': {
    'layers_num': 3,
    1: {
      'filters': [[3, 1, spec.choice(range(60, 70))],
                  [1, 3, spec.choice(range(60, 70))]],
      'activation': spec.choice(activations),
      'down_sample': {
        'size': [2, 2],
        'pooling': spec.choice(poolings),
      },
      'residual': spec.random_bit(),
      'dropout': spec.uniform(0.85, 1.0),
    },
    2: {
      'filters': [[3, 1, spec.choice(range(120, 180))],
                  [1, 3, spec.choice(range(120, 180))]],
      'activation': spec.choice(activations),
      'down_sample': {
        'size': [2, 2],
        'pooling': spec.choice(poolings),
      },
      'residual': spec.random_bit(),
      'dropout': spec.uniform(0.75, 1.0),
    },
    3: {
      'filters': [[3, 1, spec.choice(range(180, 257))],
                  [1, 3, spec.choice(range(180, 257))]],
      'activation': spec.choice(activations),
      'down_sample': {
        'size': [2, 2],
        'pooling': spec.choice(poolings),
      },
      'residual': spec.random_bit(),
      'dropout': spec.uniform(0.7, 1.0),
    }
  },

  'reduce': {
    'down_sample': {
      'pooling': spec.choice(poolings),
    }
  },

  'fc': {
    'size': spec.choice(range(400, 641)),
    'activation': spec.choice(activations),
    'dropout': spec.uniform(0.45, 1.0),
  },
}


augment_spec = {
  'scale': [1.0, spec.uniform(0.9, 1.3)],
  'crop_size': spec.choice(range(2)),
  'brightness': [spec.uniform(0.6, 1.0), spec.uniform(1.0, 1.4)],
  'contrast': [spec.uniform(0.5, 1.0), spec.uniform(1.0, 2.0)],
}
