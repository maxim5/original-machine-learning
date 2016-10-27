#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
import matplotlib.pyplot as plt
from tflearn.datasets.mnist import read_data_sets
from tflearn import ImageAugmentation

from data_set import Data, DataSet
from mnist import plot_images


def get_mnist_data():
  tf_data_sets = read_data_sets("../../../dat/mnist-tf", one_hot=False)
  convert = lambda data_set: DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
  return Data(train=convert(tf_data_sets.train),
              validation=convert(tf_data_sets.validation),
              test=convert(tf_data_sets.test))

augmentation = ImageAugmentation()
augmentation.add_random_rotation(max_angle=15)
augmentation.add_random_blur(sigma_max=0.1)
augmentation.add_random_crop(crop_shape=(28, 28), padding=4)

data = get_mnist_data()
x, y = data.train.next_batch(40)

z = np.array(x)
z = augmentation.apply(z)

plot_images(data=(x, y, y), destination=None)
plot_images(data=(z, y, y), destination=None)
plt.show()
