#! /usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from image_classification.tf.spec import *
from image_classification.tf.spec.parsed_spec import ParsedSpec

__author__ = "maxim"


class SpecTest(unittest.TestCase):
  def test_zero_nodes(self):
    def check_zero_nodes(spec):
      parsed = ParsedSpec(spec)
      self.assertEqual(parsed.size(), 0)
      self.assertEqual(spec, parsed.instantiate([]))
    
    check_zero_nodes(1)
    check_zero_nodes([])
    check_zero_nodes([1, 2, 3])
    check_zero_nodes((1, 2, 3))
    check_zero_nodes({})
    check_zero_nodes({'a': 0, 'b': 1})
    check_zero_nodes({'a': [1, 2], 'b': {'key': (1, 2)}})


  def test_uniform(self):
    spec = uniform()
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(0.5, parsed.instantiate([0.5]))
    self.assertEqual(1.0, parsed.instantiate([1.0]))


  def test_choice(self):
    spec = choice([10, 20, 30])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(10, parsed.instantiate([0.0]))
    self.assertEqual(20, parsed.instantiate([0.5]))
    self.assertEqual(30, parsed.instantiate([1.0]))


  def test_merge(self):
    spec = merge(lambda x, y: x+y, uniform(), uniform())
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(0.5, parsed.instantiate([0.0, 0.5]))
    self.assertEqual(1.5, parsed.instantiate([0.5, 1.0]))
    self.assertEqual(2.0, parsed.instantiate([1.0, 1.0]))


  def test_transform(self):
    spec = wrap(uniform(), lambda x: x*x)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(4.0, parsed.instantiate([2.0]))


  def test_transform_merge(self):
    spec = wrap(merge(lambda x, y: x+y, uniform(), uniform()), lambda x: x*x)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual(1.0, parsed.instantiate([0.0, 1.0]))
    self.assertEqual(4.0, parsed.instantiate([1.0, 1.0]))


  def test_duplicate_nodes(self):
    node = uniform()
    spec = merge(lambda x, y, z: x+y+z, node, node, node)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(3.0, parsed.instantiate([1.0]))
    self.assertEqual(9.0, parsed.instantiate([3.0]))


  def test_merge_choice(self):
    spec = choice([uniform(0, 1), uniform(2, 3)])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(0.0, parsed.instantiate([0.0, 0.0, 0.0]))
