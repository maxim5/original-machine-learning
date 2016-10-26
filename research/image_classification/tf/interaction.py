#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import os
import pathlib2
import sys

from log import log


def list_models(directory):
  result = {}
  models = os.listdir(directory)
  log('Found models:')
  for idx, model in enumerate(sorted(models)):
    path = os.path.join(directory, model, 'results.xjson')
    contents = pathlib2.Path(path).read_text()
    log('  %2d: %s -> %s' % (idx+1, model, contents))
    result[str(idx+1)] = model
  return result


def read_model(directory):
  index = list_models(directory)
  log('Your choice [default=1]')
  choice = sys.stdin.readline()
  choice = choice.strip() or '1'
  model = index.get(choice)
  assert model is not None
  return model
