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


  def test_uniform_rev(self):
    spec = uniform(4, 0)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(0.0, parsed.instantiate([0.0]))
    self.assertEqual(2.0, parsed.instantiate([0.5]))
    self.assertEqual(4.0, parsed.instantiate([1.0]))


  def test_uniform_negative(self):
    spec = uniform(-4, -2)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(-4.0, parsed.instantiate([0.0]))
    self.assertEqual(-3.0, parsed.instantiate([0.5]))
    self.assertEqual(-2.0, parsed.instantiate([1.0]))


  def test_uniform_negative_rev(self):
    spec = uniform(-2, -4)
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(-4.0, parsed.instantiate([0.0]))
    self.assertEqual(-3.0, parsed.instantiate([0.5]))
    self.assertEqual(-2.0, parsed.instantiate([1.0]))


  def test_choice(self):
    spec = choice([10, 20, 30])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(10, parsed.instantiate([0.0]))
    self.assertEqual(20, parsed.instantiate([0.5]))
    self.assertEqual(30, parsed.instantiate([1.0]))


  def test_choice_str(self):
    spec = choice(['foo', 'bar'])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual('foo', parsed.instantiate([0.0]))
    self.assertEqual('bar', parsed.instantiate([1.0]))


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
    self.assertEqual(1.0, parsed.instantiate([1.0, 0.0, 0.0]))
    self.assertEqual(2.0, parsed.instantiate([0.0, 0.0, 0.9]))
    self.assertEqual(3.0, parsed.instantiate([0.0, 1.0, 0.9]))


  def test_if_condition(self):
    def if_cond(switch, size, num):
      if switch > 0.5:
        return [size, num, num]
      return [size, num]

    spec = merge(if_cond, uniform(0, 1), uniform(1, 2), uniform(2, 3))
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)

    self.assertEqual([1, 2],    parsed.instantiate([0, 0, 0]))
    self.assertEqual([2, 3],    parsed.instantiate([0, 1, 1]))
    self.assertEqual([1, 2, 2], parsed.instantiate([1, 0, 0]))
    self.assertEqual([2, 3, 3], parsed.instantiate([1, 1, 1]))


  def test_object(self):
    class Dummy: pass
    dummy = Dummy
    dummy.value = uniform()
    dummy.foo = 'bar'
    dummy.ref = dummy

    spec = dummy
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)

    instance = parsed.instantiate([0])
    self.assertEqual(0, instance.value)
    self.assertEqual('bar', instance.foo)
    self.assertEqual(instance, instance.ref)


  def test_dict(self):
    spec = {1: uniform(), 2: choice(['foo', 'bar']), 3: merge(lambda x: -x, uniform())}
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual({1: 0.0, 2: 'foo', 3:  0.0}, parsed.instantiate([0, 0, 0]))
    self.assertEqual({1: 1.0, 2: 'bar', 3: -1.0}, parsed.instantiate([1, 1, 1]))


  def test_math_operations_1(self):
    spec = uniform() + 1
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 1)
    self.assertEqual(2.0, parsed.instantiate([1.0]))


  def test_math_operations_2(self):
    spec = uniform() * (uniform() ** 2 + 1) / uniform()
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(2.0, parsed.instantiate([1.0, 1.0, 1.0]))
    self.assertEqual(1.0, parsed.instantiate([0.5, 1.0, 1.0]))
    self.assertEqual(1.0, parsed.instantiate([0.5, 0.0, 0.5]))


  def test_math_operations_3(self):
    spec = 2 / (1 + uniform()) * (3 - uniform() + 4 ** uniform())
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 3)
    self.assertEqual(6.0, parsed.instantiate([1.0, 1.0, 1.0]))


  def test_math_operations_4(self):
    spec = choice(['foo', 'bar']) + '-' + choice(['abc', 'def'])
    parsed = ParsedSpec(spec)
    self.assertEqual(parsed.size(), 2)
    self.assertEqual('foo-abc', parsed.instantiate([0.0, 0.0]))
    self.assertEqual('bar-def', parsed.instantiate([1.0, 1.0]))
