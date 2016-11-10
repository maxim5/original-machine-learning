#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


class BaseNode(object):
  def __init__(self):
    super(BaseNode, self).__init__()
    self._point = None
    self._domain_value = None

  def value(self):
    return self._domain_value

  def set_point(self, point):
    self._point = point
    self._domain_value = self.to_domain_value(point)

  def to_domain_value(self, point):
    raise NotImplementedError()


class UniformNode(BaseNode):
  def __init__(self, start=0.0, end=1.0):
    super(UniformNode, self).__init__()
    self.shift = start
    self.scale = end - start

  def to_domain_value(self, point):
    return point * self.scale + self.shift


class ChoiceNode(BaseNode):
  def __init__(self, array):
    super(ChoiceNode, self).__init__()
    self.array = array

  def to_domain_value(self, point):
    return self.array[int(point * len(self.array))]


class MergeNode(BaseNode):
  def __init__(self, function, *children):
    super(MergeNode, self).__init__()
    assert callable(function)
    self.function = function
    self.children = children

  def value(self):
    if self._domain_value is None:
      self._domain_value = self.function(*[child.value() for child in self.children])
    return self._domain_value

  def to_domain_value(self, point):
    raise NotImplementedError()

  def set_point(self, point):
    raise NotImplementedError()
