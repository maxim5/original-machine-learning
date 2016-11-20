#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "maxim"


import numpy as np
import matplotlib.pyplot as plt
from tflearn.datasets.mnist import read_data_sets

from augmentor import ImageAugmentationPlus
from data_set import Data, DataSet
from mnist import plot_images


def get_mnist_data():
  tf_data_sets = read_data_sets("../../../dat/mnist-tf", one_hot=False)
  convert = lambda data_set: DataSet(data_set.images.reshape((-1, 28, 28, 1)), data_set.labels)
  return Data(train=convert(tf_data_sets.train),
              validation=convert(tf_data_sets.validation),
              test=convert(tf_data_sets.test))


def filter_just(x, y, value):
  return zip(*filter(lambda (_, yy): yy == value, zip(x, y)))


def plot_one(image, label=None):
  plt.figure()
  if label:
    plt.title('Label: %d' % label)
  print image.shape
  plt.imshow(image.reshape(image.shape[:-1]), cmap=plt.cm.gray_r)
  plt.draw()


########################################################################################################################


augmentation = ImageAugmentationPlus()
# augmentation.add_random_scale(downscale_limit=(1.6, 1.6), upscale_limit=(1.3, 1.3))
# augmentation.add_random_crop(crop_shape=(28, 28), padding=2)
# augmentation.add_random_rotation(max_angle=15)
# augmentation.add_random_blur(sigma_max=0.1)
augmentation.add_random_swirl(strength_limit=1.0, radius_limit=50)


def experiment(x0):
  import skimage.transform
  import skimage.util
  x1 = skimage.transform.rescale(x0, scale=(0.8, 1), preserve_range=True)
  x2 = skimage.util.pad(x1, pad_width=((3, 3), (0, 0), (0, 0)), mode='constant')

  plot_one(x0)
  plot_one(x1)
  plot_one(x2)

def experiment2():
  import matplotlib.pyplot as plt

  from skimage import data
  from skimage.transform import swirl

  image = data.checkerboard()
  swirled = swirl(image, rotation=0, strength=0.5, radius=180)

  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                 sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})

  ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
  ax0.axis('off')
  ax1.imshow(swirled, cmap=plt.cm.gray, interpolation='none')
  ax1.axis('off')

  plt.show()

########################################################################################################################

data = get_mnist_data()
data.train.just_completed = True
x, y = data.train.next_batch(36)
# x, y = filter_just(x, y, 8)

z = np.array(x)
z = augmentation.apply(z)

plot_images(data=(x, y, y), destination=None)
plot_images(data=(z, y, y), destination=None)
plt.show()


# Image warp:
# http://stackoverflow.com/questions/5071063/is-there-a-library-for-image-warping-image-morphing-for-python-with-controlled
