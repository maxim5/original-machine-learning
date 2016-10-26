#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import math

import numpy as np
import matplotlib.pyplot as plt
from tflearn.datasets.mnist import read_data_sets
from tflearn import ImageAugmentation

from data_set import Data, DataSet


def get_mnist_data():
  tf_data_sets = read_data_sets("../../../dat/mnist-tf", one_hot=False)
  convert = lambda data_set: DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
  return Data(train=convert(tf_data_sets.train),
              validation=convert(tf_data_sets.validation),
              test=convert(tf_data_sets.test))


def display(images, labels):
  num = len(images)
  rows = int(math.sqrt(num))
  cols = num / rows

  f, axes = plt.subplots(rows, cols, figsize=(rows*2,cols*2))
  axes = axes.reshape(-1)
  for i in range(len(axes)):
    a = axes[i]
    a.imshow(images[i].reshape((28, 28)), cmap=plt.cm.gray_r)
    a.set_title(labels[i])
    a.set_xticks(())
    a.set_yticks(())


augmentation = ImageAugmentation()
augmentation.add_random_rotation(max_angle=15)
augmentation.add_random_blur(sigma_max=0.1)
augmentation.add_random_crop(crop_shape=(28, 28), padding=4)

data = get_mnist_data()
x, y = data.train.next_batch(40)

z = np.array(x)
z = augmentation.apply(z)

display(x, y)
display(z, y)

plt.show()
