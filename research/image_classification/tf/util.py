#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import ast
import collections


def dict_to_str(d):
  def smart_str(val):
    if type(val) == float:
      return "%.6f" % val if val > 1e-6 else "%e" % val
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
