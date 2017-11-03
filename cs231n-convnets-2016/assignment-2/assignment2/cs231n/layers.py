import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  x_reshaped = np.reshape(x, (x.shape[0], -1))
  out = x_reshaped.dot(w) + b
  cache = (x, x_reshaped, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, x_reshaped, w, b = cache
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x_reshaped.T.dot(dout)
  db = np.sum(dout, axis=0)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(x, 0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout * (x > 0)
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from mini-batch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    mu = np.sum(x, axis=0) / N           # np.mean(x)
    x_mu = x - mu
    x_mu_sq = x_mu ** 2
    var = np.sum(x_mu_sq, axis=0) / N    # np.var(x)
    stdev = np.sqrt(var + eps)
    stdev_inv = 1.0 / stdev
    x_hat = x_mu * stdev_inv
    gamma_x_hat = gamma * x_hat
    x_bar = gamma_x_hat + beta
    out = x_bar

    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var

    cache = (mu, x_mu, x_mu_sq, var, stdev, stdev_inv, x_hat, gamma_x_hat, x_bar, beta, gamma, eps)
  elif mode == 'test':
    normalized = (x - running_mean) / np.sqrt(running_var)
    out = (normalized + beta) * gamma
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


# See https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
def batchnorm_backward(d_out, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - d_out: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - d_x: Gradient with respect to inputs x, of shape (N, D)
  - d_gamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - d_beta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  mu, x_mu, x_mu_sq, var, stdev, stdev_inv, x_hat, gamma_x_hat, x_bar, beta, gamma, eps = cache
  N, D = d_out.shape
  ones = np.ones(d_out.shape)

  # x_bar = gamma_x_hat + beta (broadcasting, beta is like a bias)
  d_gamma_x_hat = 1.0 * d_out
  d_beta = 1.0 * np.sum(d_out, axis=0)

  # gamma_x_hat = gamma * x_hat (broadcasting, not a dot product)
  d_x_hat = gamma * d_gamma_x_hat
  d_gamma = np.sum(x_hat * d_gamma_x_hat, axis=0)

  # x_hat = x_mu * stdev_inv (the same: broadcasting, not a dot product)
  d1_x_mu = stdev_inv * d_x_hat
  d_stdev_inv = np.sum(x_mu * d_x_hat, axis=0)

  # stdev_inv = 1.0 / stdev (per-element)
  d_stdev = -1.0 / (stdev**2) * d_stdev_inv

  # stdev = np.sqrt(var + eps) (per-element)
  d_var = 0.5 / stdev * d_stdev

  # var = np.sum(x_mu_sq, axis=0) / N
  d_x_mu_sq = 1.0 / N * ones * d_var

  # x_mu_sq = x_mu ** 2
  d2_x_mu = 2 * x_mu * d_x_mu_sq

  # gradients of (x-mu) from different paths join together
  d_x_mu = d1_x_mu + d2_x_mu

  # x_mu = x - mu
  d1_x = d_x_mu
  d_mu = -np.sum(d_x_mu, axis=0)

  # mean = np.sum(x, axis=0) / N
  d2_x = 1.0 / N * ones * d_mu

  # gradients of x from different paths join together
  d_x = d1_x + d2_x

  return d_x, d_gamma, d_beta

# See http://cthorey.github.io/backpropagation/
def batchnorm_backward_alt(d_out, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  mu, x_mu, x_mu_sq, var, stdev, stdev_inv, x_hat, gamma_x_hat, x_bar, beta, gamma, eps = cache
  N, D = d_out.shape

  d_beta = np.sum(d_out, axis=0)
  d_gamma = np.sum(x_hat * d_out, axis=0)
  d_x = (1.0 / N) * gamma * stdev_inv * (N * d_out - np.sum(d_out, axis=0) - x_mu * (stdev_inv**2) * np.sum(d_out * x_mu, axis=0))
  
  return d_x, d_gamma, d_beta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = np.random.random(x.shape) < 1 - p
    out = x * mask / (1 - p)
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    p = dropout_param['p']
    dx = dout * mask / (1 - p)
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, H_out, W_out))

  x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

  for n in xrange(N):
    for f in xrange(F):
      filter_w = w[f, :, :, :]                            # 3-dimensional: (C, HH, WW)
      for out_i, i in enumerate(xrange(0, H, stride)):
        for out_j, j in enumerate(xrange(0, W, stride)):
          image_patch = x_pad[n, :, i:i+HH, j:j+WW]       # 3-dimensional: (C, HH, WW)
          out[n, f, out_i, out_j] += np.sum(filter_w * image_patch)
      out[n, f, :, :] += b[f]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(d_out, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, C, H_out, W_out = d_out.shape

  x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

  db = np.sum(d_out, axis=(0, 2, 3))
  # Full version:
  #
  # db = np.zeros_like(b)
  # for f in xrange(F):
  #   db[f] = np.sum(d_out[:, f, :, :])

  dw = np.zeros_like(w)
  dx = np.zeros_like(x_pad)
  for n in xrange(N):
    for f in xrange(F):
      filter_w = w[f, :, :, :]
      for out_i, i in enumerate(xrange(0, H, stride)):
        for out_j, j in enumerate(xrange(0, W, stride)):
          dw[f, :, :, :] += d_out[n, f , out_i, out_j] * x_pad[n, :, i:i+HH, j:j+WW]
          dx[n, :, i:i+HH, j:j+WW] += filter_w * d_out[n, f, out_i, out_j]
  dx = dx[:,:,1:H+1,1:W+1]

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H, W = x.shape
  H_out = 1 + (H - pool_height) / stride
  W_out = 1 + (W - pool_width) / stride

  out = np.zeros((N, C, H_out, W_out))
  for out_i, i in enumerate(xrange(0, H, stride)):
    for out_j, j in enumerate(xrange(0, W, stride)):
      out[:, :, out_i, out_j] = np.max(x[:, :, i:i+pool_height, j:j+pool_width], axis=(2, 3))

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(d_out, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  N, C, H_out, W_out = d_out.shape

  dx = np.zeros_like(x)
  for n in xrange(N):
    for c in xrange(C):
      for out_i, i in enumerate(xrange(0, H, stride)):
        for out_j, j in enumerate(xrange(0, W, stride)):
          maxidx = np.unravel_index(np.argmax(x[n, c, i:i+pool_height, j:j+pool_width]), (pool_height, pool_width))
          dx[n, c, i+maxidx[0], j+maxidx[1]] += d_out[n, c, out_i, out_j]
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape
  x = x.transpose(0, 2, 3, 1).reshape((N * H * W, C))
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
  return out, cache


def spatial_batchnorm_backward(d_out, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = d_out.shape
  d_out = d_out.transpose(0, 2, 3, 1).reshape((N * H * W, C))
  d_x, d_gamma, d_beta = batchnorm_backward_alt(d_out, cache)
  d_x = d_x.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
  return d_x, d_gamma, d_beta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
