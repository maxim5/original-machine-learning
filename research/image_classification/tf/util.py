#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import json


def dict_to_str(d):
  def smart_str(val):
    if type(val) == float:
      return "%.5f" % val
    return repr(val)

  return '{%s}' % ', '.join(['%s: %s' % (repr(k), smart_str(d[k])) for k in sorted(d.keys())])


def str_to_dict(s):
  s = s.replace("'", "\"")
  return json.loads(s)


def zip_longest(list1, list2):
  len1 = len(list1)
  len2 = len(list2)
  for i in xrange(max(len1, len2)):
    yield (list1[i % len1], list2[i % len2])
