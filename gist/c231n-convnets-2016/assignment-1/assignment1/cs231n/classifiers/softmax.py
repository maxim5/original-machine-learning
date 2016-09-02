import numpy as np

def softmax(scores, idx):
  return np.exp(scores[idx]) / np.sum(np.exp(scores))

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]

  loss = 0.0
  dW = np.zeros_like(W)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    self_information = -np.log(softmax(scores, idx=y[i]))
    loss += self_information

    dW[:,y[i]] -= X[i,:].T
    for j in xrange(num_classes):
      dW[:,j] += X[i,:].T * softmax(scores, idx=j)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]

  # Compute the scores and correct scores in bulk (see linear_svm.py).
  # Note the normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
  scores = X.dot(W)
  scores -= np.max(scores)
  correct_scores = scores[np.arange(num_train), y]

  # Compute the softmax per correct scores in bulk, and sum over its logs.
  exponents = np.exp(scores)
  sums_per_row = np.sum(exponents, axis=1)
  softmax_array = np.exp(correct_scores) / sums_per_row
  information_array = -np.log(softmax_array)
  loss = np.mean(information_array)

  # Compute the softmax per whole scores matrix and dot product with X.
  all_softmax_matrix = (exponents.T / sums_per_row).T
  grad_coeff = np.zeros_like(scores)
  grad_coeff[np.arange(num_train), y] = -1
  grad_coeff += all_softmax_matrix
  dW = np.dot(X.T, grad_coeff) / num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
