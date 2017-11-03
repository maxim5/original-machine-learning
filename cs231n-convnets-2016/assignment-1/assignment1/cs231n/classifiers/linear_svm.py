import numpy as np

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j   ] += X[i,:].T  # the grad from class j
        dW[:,y[i]] -= X[i,:].T  # the grad from class y[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]

  # Here's a bit faster, yet still naive approach that we're going to vectorize:
  #
  # for i in xrange(num_train):
  #   scores = X[i].dot(W)
  #   correct_class_score = scores[y[i]]
  #   loss_values = np.maximum(scores - correct_class_score + 1, 0)
  #   loss_values[y[i]] = 0
  #   loss += np.sum(loss_values)

  # Compute the scores in one multiply.
  scores = X.dot(W)

  # Get the correct values in one slicing operation -> [np.arange(num_train), y].
  # Note that `correct_scores` is an array of shape `(num_train)`.
  # See advanced indexing: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  # The other option is sadly slow:
  # correct_scores = np.diag(scores[y,:])
  correct_scores = scores[np.arange(num_train), y]

  # The broadcasting rules aren't trivial.
  # Can't compute `scores - correct_scores`, but can if transpose `scores` first.
  # See http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  loss_values = (scores.T - correct_scores + 1).T
  loss_values[np.arange(num_train), y] = 0
  hinge_values = np.maximum(loss_values, 0)

  # Now it's pretty easy.
  loss = np.sum(hinge_values) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  # Compute the gradient.
  # Idea: perform the two operations simultaneously
  # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
  # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T

  # First transform `hinge_values` is a matrix of zeros and ones.
  # The values of `X` should be added to `dW` where the coefficient is one (1).
  grad_coeff = hinge_values
  grad_coeff[hinge_values > 0] = 1

  # Also the values of `X` should be subtracted from `dW` on the correct classes -> [np.arange(num_train), y] (2).
  # The number of times it's subtracted corresponds to the number of positive margins, i.e. the sum per each row.
  sums_per_row = np.sum(grad_coeff, axis=1)
  grad_coeff[np.arange(num_train), y] = -sums_per_row[np.arange(num_train)]

  # Finally add up all values of `X` with computed coefficients.
  dW = np.dot(X.T, grad_coeff)

  # Boilerplate stuff.
  dW /= num_train
  dW += reg * W

  return loss, dW
