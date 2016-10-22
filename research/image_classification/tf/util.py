#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import ast
import collections
import random
import string

import numpy as np


def dict_to_str(d):
  def smart_str(val):
    if type(val) == float:
      return "%.6f" % val if abs(val) > 1e-6 else "%e" % val
    if type(val) == dict:
      return dict_to_str(val)
    return repr(val)

  return '{%s}' % ', '.join(['%s: %s' % (repr(k), smart_str(d[k])) for k in sorted(d.keys())])


def str_to_dict(s):
  return ast.literal_eval(s)


def zip_longest(list1, list2):
  len1 = len(list1)
  len2 = len(list2)
  for i in xrange(max(len1, len2)):
    yield (list1[i % len1], list2[i % len2])


def deep_update(dict_, upd):
  for key, value in upd.iteritems():
    if isinstance(value, collections.Mapping):
      recursive = deep_update(dict_.get(key, {}), value)
      dict_[key] = recursive
    else:
      dict_[key] = upd[key]
  return dict_


def mini_batch(total, size):
  return zip(range(0, total, size),
             range(size, total + size, size))


def random_id(size=6, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in xrange(size))


def safe_concat(list_):
  list_ = [i for i in list_ if i is not None]
  if len(list_) == 0:
    return None
  if type(list_[0]) == np.ndarray:
    return np.concatenate(list_)
  return list_
