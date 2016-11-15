#! /usr/bin/env python
# -*- coding: utf-8 -*-
from image_classification.tf.bayesian.artist import Artist
from image_classification.tf.bayesian.sampler import DefaultSampler
from image_classification.tf.bayesian.strategy import BayesianStrategy
from image_classification.tf.log import log
from image_classification.tf.mnist_spec import hyper_params
from image_classification.tf.spec import ParsedSpec

__author__ = "maxim"


parsed = ParsedSpec(hyper_params)
log('Spec size=%d' % parsed.size())

sampler = DefaultSampler()
sampler.add_uniform(parsed.size())

strategy_params = {
  'io_load_dir': 'mnist-conv-3',
}
strategy = BayesianStrategy(sampler, **strategy_params)
print strategy.points
print strategy.values

artist = Artist(strategy=strategy, names=parsed.get_names())
artist.scatter_plot_per_dimension()
