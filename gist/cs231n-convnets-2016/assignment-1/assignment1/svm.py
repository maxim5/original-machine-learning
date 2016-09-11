#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "maxim"

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers import LinearSVM

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_data():
    from cs231n.data_utils import load_CIFAR10

    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    plot_dataset(X_train, y_train)

    num_training = 49000
    num_validation = 1000
    num_test = 1000

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print '[Reshaped] Training data shape: ', X_train.shape, y_train.shape
    print '[Reshaped] Validation data shape: ', X_val.shape, y_val.shape
    print '[Reshaped] Test data shape: ', X_test.shape, y_test.shape

    return X_train, y_train, X_val, y_val, X_test, y_test


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


def norm_data(X_train, X_val, X_test):
    # Pre-processing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)

    plt.figure(figsize=(4,4))
    plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
    plt.show()

    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    print '[Normalized] Training data shape: ', X_train.shape, y_train.shape
    print '[Normalized] Validation data shape: ', X_val.shape, y_val.shape
    print '[Normalized] Test data shape: ', X_test.shape, y_test.shape

    return X_train, X_val, X_test


def train(X_train, y_train, X_val, y_val):
    # Use the validation set to tune hyperparameters (regularization strength and
    # learning rate). You should experiment with different ranges for the learning
    # rates and regularization strengths; if you are careful you should be able to
    # get a classification accuracy of about 0.4 on the validation set.
    learning_rates = [1e-7, 5e-6, 1e-6]
    regularization_strengths = [1e4, 5e4, 1e5]

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1   # The highest validation accuracy that we have seen so far.
    best_svm = None # The LinearSVM object that achieved the highest validation rate.

    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = LinearSVM()
            svm.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1000)

            y_train_pred = svm.predict(X_train)
            train_accuracy = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)

            results[(lr, reg)] = (train_accuracy, val_accuracy)

            if best_val < val_accuracy:
                best_val = val_accuracy
                best_svm = svm

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'Learning-rate=%f regularizer=%f train-accuracy=%f validation-accuracy=%f' % (lr, reg, train_accuracy, val_accuracy)

    print 'Best validation accuracy achieved during cross-validation: %f' % best_val
    return best_svm


def train_improved(X_train, y_train, X_val, y_val, max_epochs=50):
    class Holder(object): pass
    state = Holder()
    state.accuracy = -1
    state.svm = None
    state.hyper = None
    state.epoch = 0

    def compute_at(hyper_params):
        learning_rate, regularizer = hyper_params
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=learning_rate, reg=regularizer, num_iters=1000)

        y_train_prediction = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_prediction)
        y_val_prediction = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_prediction)
        final_accuracy = min(train_accuracy, val_accuracy)

        state.epoch += 1
        improved = state.accuracy < final_accuracy
        if improved:
            state.accuracy = final_accuracy
            state.svm = svm
            state.hyper = hyper_params[:]

        print "Epoch %2d: (%.8f, %f) -> %f %s" % (state.epoch, learning_rate, regularizer, final_accuracy, "(!)" if improved else "")
        return improved, final_accuracy

    learning_rate = 1e-4
    regularizer = 1e4
    hyper_params = [learning_rate, regularizer]
    hyper_size = len(hyper_params)

    while state.epoch < max_epochs:
        if state.accuracy > 0:
            idx = np.random.randint(hyper_size)
            updated_params = hyper_params[:]
            choice = np.random.rand()
            if choice < 0.1:
                updated_params[idx] = hyper_params[idx] * (20 * np.random.rand())
            if choice < 0.2:
                updated_params[idx] = hyper_params[idx] / (20 * np.random.rand())
            else:
                updated_params[idx] += hyper_params[idx] * (2 * np.random.rand() - 1)
        else:
            updated_params = hyper_params

        improved, _ = compute_at(updated_params)
        if improved:
            hyper_params = updated_params

    print "Best: (%.8f, %f) -> %f" % (state.hyper[0], state.hyper[1], state.accuracy)
    return state.svm


def test(svm, X_test, y_test):
    y_test_pred = svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print 'Linear SVM on raw pixels final test set accuracy: %f' % test_accuracy


def plot_weights(svm):
    w = svm.W[:-1,:]    # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

X_train, y_train, X_val, y_val, X_test, y_test = get_data()
X_train, X_val, X_test = norm_data(X_train, X_val, X_test)
# svm = train(X_train, y_train, X_val, y_val)
svm = train_improved(X_train, y_train, X_val, y_val)
test(svm, X_test, y_test)
plot_weights(svm)
