#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def get_data():
    from cs231n.data_utils import load_CIFAR10

    # Load the raw CIFAR-10 data.
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print 'Training data shape: ', X_train.shape
    print 'Training labels shape: ', y_train.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    plot_dataset(X_train, y_train)

    # Subsample the data for more efficient code execution in this exercise
    num_training = 5000
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print 'Reshaped X: ', X_train.shape, X_test.shape

    return X_train, y_train, X_test, y_test


def plot_dataset(X_train, y_train):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def cross_validate(X_train, y_train, num_folds=5):
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {k:[] for k in k_choices}

    for i in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[:i] + X_train_folds[i+1:])
        y_train_cv = np.hstack(y_train_folds[:i] + y_train_folds[i+1:])

        X_val = X_train_folds[i]
        y_val = y_train_folds[i]

        classifier = KNearestNeighbor()
        classifier.train(X_train_cv, y_train_cv)
        dists_cv = classifier.compute_distances_no_loops(X_val)

        for k in k_choices:
            y_val_pred = classifier.predict_labels(dists_cv, k=k)
            num_correct = np.sum(y_val_pred == y_val)
            accuracy = float(num_correct) / len(y_val)
            k_to_accuracies[k].append(accuracy)

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print 'k = %d, accuracy = %f' % (k, accuracy)

    plot_cross_validation(k_choices, k_to_accuracies)

    best_accuracy = sorted(k_to_accuracies, key=lambda k: np.mean(k_to_accuracies[k]))
    return k_choices[best_accuracy[0]]


def plot_cross_validation(k_choices, k_to_accuracies):
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


def test(X_train, y_train, X_test, y_test, best_k):
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    # Compute and display the accuracy
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / len(y_test)
    print 'Best k=%d' % best_k
    print 'Got %d / %d correct => accuracy: %f' % (num_correct, len(y_test), accuracy)


X_train, y_train, X_test, y_test = get_data()
best_k = cross_validate(X_train, y_train)
test(X_train, y_train, X_test, y_test, best_k)
