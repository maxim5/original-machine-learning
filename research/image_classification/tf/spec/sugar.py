#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import math
from scipy import stats

from nodes import *


def wrap(node, transform):
  if transform is not None:
    return MergeNode(transform, node)
  return node

def uniform(start=0.0, end=1.0, transform=None):
  node = UniformNode(start, end)
  return wrap(node, transform)

def normal(mean=0.0, stdev=1.0):
  return NonUniformNode(ppf=stats.norm.ppf, loc=mean, scale=stdev)

def choice(array, transform=None):
  if not [item for item in array if isinstance(item, BaseNode)]:
    node = ChoiceNode(*array)
  else:
    node = MergeChoiceNode(*array)
  return wrap(node, transform)

def merge(function, *nodes):
  return MergeNode(function, *nodes)

def exp(node): return merge(math.exp, node)
def expm1(node): return merge(math.expm1, node)
def frexp(node): return merge(math.frexp, node)
def ldexp(node, i): return merge(lambda x: math.ldexp(x, i), node)

def sqrt(node): return merge(math.sqrt, node)
def pow(a, b): return a ** b

def log(node, base=None): return merge(lambda x: math.log(x, base), node)
def log1p(node): return merge(math.log1p, node)
def log10(node): return merge(math.log10, node)

def sin(node): return merge(math.sin, node)
def cos(node): return merge(math.cos, node)
def tan(node): return merge(math.tan, node)

def sinh(node): return merge(math.sinh, node)
def cosh(node): return merge(math.cosh, node)
def tanh(node): return merge(math.tanh, node)

def asin(node): return merge(math.asin, node)
def acos(node): return merge(math.acos, node)
def atan(node): return merge(math.atan, node)
def atan2(node): return merge(math.atan2, node)

def asinh(node): return merge(math.asinh, node)
def acosh(node): return merge(math.acosh, node)
def atanh(node): return merge(math.atanh, node)

def min_(*array):
  nodes = [item for item in array if isinstance(item, BaseNode)]
  if len(nodes) == 0:
    return min(*array) if len(array) > 1 else array[0]
  node = merge(min, *nodes) if len(nodes) > 1 else nodes[0]

  rest = [item for item in array if not isinstance(item, BaseNode)]
  if rest:
    node = merge(lambda x: min(x, *rest), node)
  return node

def max_(*array):
  nodes = [item for item in array if isinstance(item, BaseNode)]
  if len(nodes) == 0:
    return max(*array) if len(array) > 1 else array[0]
  node = merge(max, *nodes) if len(nodes) > 1 else nodes[0]

  rest = [item for item in array if not isinstance(item, BaseNode)]
  if rest:
    node = merge(lambda x: max(x, *rest), node)
  return node
