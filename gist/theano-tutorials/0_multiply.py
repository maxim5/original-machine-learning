#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import theano
from theano import tensor as T

a = T.scalar()
b = T.scalar()
y = a * b

multiply = theano.function(inputs=[a, b], outputs=y)

print multiply(1, 2)
print multiply(3, 3)
