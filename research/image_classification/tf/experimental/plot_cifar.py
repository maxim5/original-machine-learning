#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'maxim'


import math
import numpy as np
import matplotlib.pyplot as plt

from image_classification.tf.augmentor import ImageAugmentationPlus
from image_classification.tf.cifar10.main import get_cifar10_data


def plot_images(images, labels, destination):
  num = min(len(images), 100)
  rows = int(math.sqrt(num))
  cols = (num + rows - 1) / rows

  if destination:
    plt.figure()

  f, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2), dpi=80, facecolor='w', edgecolor='k')
  axes = axes.reshape(-1)
  for i in xrange(len(axes)):
    if i < len(images):
      ax = axes[i]
      for spine in ax.spines.values():
        spine.set_edgecolor('gray')
      ax.imshow(images[i].reshape((32, 32, 3)), interpolation='nearest', aspect=1)
      ax.set_title('label: %d' % labels[i])
      ax.set_xticks(())
      ax.set_yticks(())
    else:
      axes[i].axis('off')

  if destination:
    destination += '.png'
    plt.savefig(destination, bbox_inches='tight')
    plt.close()
    return destination

augmentation = ImageAugmentationPlus()
augmentation.add_random_flip_leftright()
augmentation.add_random_scale(downscale_limit=(1.0, 1.0), upscale_limit=(1.3, 1.3), fix_aspect_ratio=True)
augmentation.add_random_crop(crop_shape=(32, 32), padding=3)
augmentation.add_random_brightness(0.7, 1.4)
augmentation.add_random_contrast(0.5, 2.0)
# augmentation.add_random_rotation(max_angle=10)
# augmentation.add_random_blur(sigma_max=0.2)
# augmentation.add_random_swirl(strength_limit=1.0, radius_limit=100)


data = get_cifar10_data(one_hot=False)
data.train.just_completed = True
x, y = data.train.next_batch(36)

z = np.array(x)
z = augmentation.apply(z)

plot_images(x, y, destination=None)
plot_images(z, y, destination=None)
plt.show()
