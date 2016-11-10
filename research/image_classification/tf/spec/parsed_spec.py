#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import copy
import numbers

from nodes import BaseNode, MergeNode


class ParsedSpec(object):
  def __init__(self, spec):
    self._spec = spec
    self._leaves = {}
    self._traverse_index = -1
    self._traverse_leaves_recursive(spec)

  def _traverse_leaves_recursive(self, spec):
    if isinstance(spec, BaseNode):
      node = spec

      if isinstance(node, MergeNode):
        for child in node.children:
          self._traverse_leaves_recursive(child)
        return

      if not node in self._leaves:
        self._traverse_index += 1
        self._leaves[node] = self._traverse_index

      return

    if isinstance(spec, numbers.Number):
      return

    if isinstance(spec, dict):
      for key, value in spec.iteritems():
        self._traverse_leaves_recursive(value)
      return

    if isinstance(spec, list) or isinstance(spec, tuple):
      for item in spec:
        self._traverse_leaves_recursive(item)
      return

    if isinstance(spec, object):
      for key, value in spec.__dict__.iteritems():
        self._traverse_leaves_recursive(value)
      return

  def instantiate(self, points):
    for node, index in self._leaves.iteritems():
      node.set_point(points[index])

    spec_copy = copy.deepcopy(self._spec)
    self._traverse_and_replace(spec_copy)
    return spec_copy

  def _traverse_and_replace(self, spec_copy):
    if isinstance(spec_copy, BaseNode):
      return spec_copy.value()

    if isinstance(spec_copy, numbers.Number):
      return spec_copy

    if isinstance(spec_copy, dict):
      for key, value in spec_copy.iteritems():
        spec_copy[key] = self._traverse_and_replace(spec_copy[key])
      return spec_copy

    if isinstance(spec_copy, list) or isinstance(spec_copy, tuple):
      return [self._traverse_and_replace(item_copy) for item_copy in spec_copy]

    if isinstance(spec_copy, object):
      for key, value in spec_copy.__dict__.iteritems():
        setattr(copy, key, self._traverse_and_replace(spec_copy))
      return spec_copy

    return spec_copy
