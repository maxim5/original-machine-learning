import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    pooled_height = 1 + (H - 2) / 2
    pooled_width = 1 + (W - 2) / 2
    self.params["W1"] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params["b1"] = np.zeros(num_filters)
    self.params["W2"] = np.random.randn(num_filters * pooled_height * pooled_width, hidden_dim) * weight_scale
    self.params["b2"] = np.zeros(hidden_dim)
    self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b3"] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    forward_msg = X
    forward_msg, conv_relu_cache = conv_relu_forward(forward_msg, W1, b1, conv_param)
    forward_msg, max_pool_cache = max_pool_forward_fast(forward_msg, pool_param)
    forward_msg, affine_relu_cache = affine_relu_forward(forward_msg, W2, b2)
    forward_msg, fc_cache = affine_forward(forward_msg, W3, b3)
    if y is None:
      return forward_msg
    loss, backward_msg = softmax_loss(forward_msg, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

    backward_msg, dW3, db3 = affine_backward(backward_msg, fc_cache)
    backward_msg, dW2, db2 = affine_relu_backward(backward_msg, affine_relu_cache)
    backward_msg = max_pool_backward_fast(backward_msg, max_pool_cache)
    backward_msg, dW1, db1 = conv_relu_backward(backward_msg, conv_relu_cache)

    grads = {
      "W1": dW1 + self.reg * W1,
      "b1": db1,
      "W2": dW2 + self.reg * W2,
      "b2": db2,
      "W3": dW3 + self.reg * W3,
      "b3": db3,
    }

    return loss, grads


class MultiLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  [conv - relu - max-pooling - batch-norm]* - affine - relu - batch-norm - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_classes=10,
               conv_layers=((32, (3, 3, 1, 1), (2, 2, 2)), (32, (3, 3, 1, 1), (2, 2, 2))), hidden_dim=100,
               weight_scale=1e-3, reg=0.0, bn_params={'eps': 1e-5, 'momentum': 0.9}, dtype=np.float32):
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    height = H
    width = W
    channels = C

    self.params = {}
    self.extras = {}
    for i, conv_param in enumerate(conv_layers):
      num_filters, (filter_size_h, filter_size_w, stride, padding), (pool_height, pool_width, pool_stride) = conv_param

      self.extras["conv%d" % (i + 1)] = {'stride': stride, 'pad': padding}
      self.extras["pool%d" % (i + 1)] = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}
      self.extras["bn%d" % (i + 1)] = bn_params.copy()

      self.params["W%d" % (i + 1)] = np.random.randn(num_filters, channels, filter_size_h, filter_size_w) * weight_scale
      self.params["b%d" % (i + 1)] = np.zeros(num_filters)
      self.params["beta%d" % (i + 1)] = np.zeros(num_filters)
      self.params["gamma%d" % (i + 1)] = np.ones(num_filters)

      # convolution
      height = 1 + (height - filter_size_h + 2 * padding) / stride
      width = 1 + (width - filter_size_w + 2 * padding) / stride
      # pooling
      height = 1 + (height - pool_height) / pool_stride
      width = 1 + (width - pool_width) / pool_stride

      channels = num_filters

    # Extra hidden "affine - relu - batch-norm"
    num_conv_layers = len(conv_layers)
    self.num_conv_layers = num_conv_layers
    self.extras["bn%d" % (num_conv_layers + 1)] = bn_params.copy()
    self.params["W%d" % (num_conv_layers + 1)] = np.random.randn(channels * height * width, hidden_dim) * weight_scale
    self.params["b%d" % (num_conv_layers + 1)] = np.zeros(hidden_dim)
    self.params["beta%d" % (num_conv_layers + 1)] = np.zeros(hidden_dim)
    self.params["gamma%d" % (num_conv_layers + 1)] = np.ones(hidden_dim)

    # Output fully-connected layer
    self.params["W%d" % (num_conv_layers + 2)] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b%d" % (num_conv_layers + 2)] = np.zeros(num_classes)


  def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    forward_msg = X
    conv_layers_caches = []
    for i in xrange(1, self.num_conv_layers + 1):
      W = self.params["W%d" % i]
      b = self.params["b%d" % i]
      conv_param = self.extras["conv%d" % i]
      pool_param = self.extras["pool%d" % i]
      beta = self.params["beta%d" % i]
      gamma = self.params["gamma%d" % i]
      bn_params = self.extras["bn%d" % i]
      bn_params['mode'] = mode

      forward_msg, conv_relu_cache = conv_relu_forward(forward_msg, W, b, conv_param)
      forward_msg, max_pool_cache = max_pool_forward_fast(forward_msg, pool_param)
      forward_msg, bn_cache = spatial_batchnorm_forward(forward_msg, gamma, beta, bn_params)

      cache = (conv_relu_cache, max_pool_cache, bn_cache)
      conv_layers_caches.append(cache)

    # Extra "affine - relu - batch-norm"
    W = self.params["W%d" % (self.num_conv_layers + 1)]
    b = self.params["b%d" % (self.num_conv_layers + 1)]
    beta = self.params["beta%d" % (self.num_conv_layers + 1)]
    gamma = self.params["gamma%d" % (self.num_conv_layers + 1)]
    bn_params = self.extras["bn%d" % (self.num_conv_layers + 1)]
    bn_params['mode'] = mode

    forward_msg, affine_relu_cache = affine_relu_forward(forward_msg, W, b)
    forward_msg, bn_cache = batchnorm_forward(forward_msg, gamma, beta, bn_params)

    W = self.params["W%d" % (self.num_conv_layers + 2)]
    b = self.params["b%d" % (self.num_conv_layers + 2)]
    forward_msg, fc_cache = affine_forward(forward_msg, W, b)

    if mode == 'test':
      return forward_msg

    loss, backward_msg = softmax_loss(forward_msg, y)
    for i in xrange(1, self.num_conv_layers + 3):
      W = self.params["W%d" % i]
      loss += 0.5 * self.reg * np.sum(W * W)

    # Backward
    grads = {}
    backward_msg, dW, db = affine_backward(backward_msg, fc_cache)
    grads["W%d" % (self.num_conv_layers + 2)] = dW + self.reg * self.params["W%d" % (self.num_conv_layers + 2)]
    grads["b%d" % (self.num_conv_layers + 2)] = db

    backward_msg, dgamma, dbeta = batchnorm_backward_alt(backward_msg, bn_cache)
    grads["beta%d" % (self.num_conv_layers + 1)] = dbeta
    grads["gamma%d" % (self.num_conv_layers + 1)] = dgamma

    backward_msg, dW, db = affine_relu_backward(backward_msg, affine_relu_cache)
    grads["W%d" % (self.num_conv_layers + 1)] = dW + self.reg * self.params["W%d" % (self.num_conv_layers + 1)]
    grads["b%d" % (self.num_conv_layers + 1)] = db

    for i in xrange(self.num_conv_layers, 0, -1):
      conv_relu_cache, max_pool_cache, bn_cache = conv_layers_caches[i - 1]

      backward_msg, dgamma, dbeta = spatial_batchnorm_backward(backward_msg, bn_cache)
      grads["beta%d" % i] = dbeta
      grads["gamma%d" % i] = dgamma

      backward_msg = max_pool_backward_fast(backward_msg, max_pool_cache)
      backward_msg, dW, db = conv_relu_backward(backward_msg, conv_relu_cache)
      grads["W%d" % i] = dW + self.reg * self.params["W%d" % i]
      grads["b%d" % i] = db

    return loss, grads
