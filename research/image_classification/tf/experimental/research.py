#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'

from image_classification.tf.bayesian.artist import Artist
from image_classification.tf.bayesian.sampler import DefaultSampler
from image_classification.tf.bayesian.strategy import BayesianStrategy
from image_classification.tf.cifar10.cifar_spec import *
from image_classification.tf.logging import log
from image_classification.tf.spec import ParsedSpec

parsed = ParsedSpec(hyper_params_spec_2_0)
log('Spec size=%d' % parsed.size())

sampler = DefaultSampler()
sampler.add_uniform(parsed.size())

strategy_params = {
  'io_load_dir': '../../_models/cifar10/hyper/stage1-2.0',
}
strategy = BayesianStrategy(sampler, **strategy_params)
print strategy.points
print strategy.values

artist = Artist(strategy=strategy, names=parsed.get_names())
artist.scatter_plot_per_dimension()
