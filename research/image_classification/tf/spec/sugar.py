#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


from nodes import *


def wrap(node, transform):
  if transform is not None:
    return MergeNode(transform, node)
  return node

def uniform(start=0.0, end=1.0, transform=None):
  node = UniformNode(start, end)
  return wrap(node, transform)

def choice(array, transform=None):
  if not [item for item in array if isinstance(item, BaseNode)]:
    node = ChoiceNode(*array)
  else:
    node = MergeChoiceNode(*array)
  return wrap(node, transform)

def merge(function, *nodes):
  return MergeNode(function, *nodes)
