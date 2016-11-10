#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


class BaseNode(object):
  def __init__(self):
    super(BaseNode, self).__init__()
    self._domain_value = None

  def value(self):
    return self._domain_value


class AcceptsInputNode(BaseNode):
  def __init__(self):
    super(AcceptsInputNode, self).__init__()
    self._point = None

  def set_point(self, point):
    self._point = point
    self._domain_value = self.to_domain_value(point)

  def to_domain_value(self, point):
    raise NotImplementedError()


class UniformNode(AcceptsInputNode):
  def __init__(self, start=0.0, end=1.0):
    super(UniformNode, self).__init__()
    self.shift = start
    self.scale = end - start

  def to_domain_value(self, point):
    return point * self.scale + self.shift


class ChoiceNode(AcceptsInputNode):
  def __init__(self, *array):
    super(ChoiceNode, self).__init__()
    self.array = array

  def to_domain_value(self, point):
    index = int(point * len(self.array))
    return self.array[min(index, len(self.array) - 1)]


class JointNode(BaseNode):
  def __init__(self, *children):
    super(JointNode, self).__init__()
    self.children = children


class MergeNode(JointNode):
  def __init__(self, function, *children):
    super(MergeNode, self).__init__(*children)
    assert callable(function)
    self.function = function

  def value(self):
    if self._domain_value is None:
      self._domain_value = self.function(*[child.value() for child in self.children])
    return self._domain_value


class MergeChoiceNode(JointNode, AcceptsInputNode):
  def __init__(self, *children):
    super(MergeChoiceNode, self).__init__(*children)

  def value(self):
    if self._domain_value is None and self._point is not None:
      values = [child.value() if isinstance(child, BaseNode) else child for child in self.children]
      index = int(self._point * len(values))
      self._domain_value = values[min(index, len(values) - 1)]
    return self._domain_value

  def to_domain_value(self, point):
    return None
