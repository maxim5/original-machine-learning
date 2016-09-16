import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {
      "W1": np.random.randn(input_dim, hidden_dim) * weight_scale,
      "b1": np.zeros(hidden_dim),
      "W2": np.random.randn(hidden_dim, num_classes) * weight_scale,
      "b2": np.zeros(num_classes),
    }
    self.reg = reg


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """

    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]

    layer1, cache1 = affine_forward(X, W1, b1)
    layer2, cache2 = relu_forward(layer1)
    layer3, cache3 = affine_forward(layer2, W2, b2)
    if y is None: return layer3
    layer4, cache4 = softmax_loss(layer3, y)

    loss = layer4 + 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)

    gradient4 = cache4
    gradient3, dW2, db2 = affine_backward(gradient4, cache3)
    gradient2 = relu_backward(gradient3, cache2)
    gradient1, dW1, db1 = affine_backward(gradient2, cache1)
    
    grads = {
      "W1": dW1 + self.reg * W1,
      "b1": db1,
      "W2": dW2 + self.reg * W2,
      "b2": db2,
    }

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    dims1 = [input_dim] + hidden_dims
    dims2 = hidden_dims + [num_classes]
    for idx, (dim1, dim2) in enumerate(zip(dims1, dims2)):
      self.params["W%d" % (idx + 1)] = np.random.randn(dim1, dim2) * weight_scale
      self.params["b%d" % (idx + 1)] = np.zeros(dim2)
      if use_batchnorm and idx < self.num_layers - 1:
        self.params["beta%d" % (idx + 1)] = np.zeros(dim2)
        self.params["gamma%d" % (idx + 1)] = np.ones(dim2)

    self.reg = reg

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    ############################################################################
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    ############################################################################

    forward_msg = X
    affine_caches = []
    dropout_caches = []
    batchnorm_caches = []
    for i in xrange(1, self.num_layers + 1):
      W = self.params["W%d" % i]
      b = self.params["b%d" % i]
      if i != self.num_layers:
        forward_msg, cache = affine_relu_forward(forward_msg, W, b)
      else:
        forward_msg, cache = affine_forward(forward_msg, W, b)
      affine_caches.append(cache)

      if self.use_dropout:
        forward_msg, cache = dropout_forward(forward_msg, self.dropout_param)
        dropout_caches.append(cache)

      if i != self.num_layers and self.use_batchnorm:
        beta = self.params["beta%d" % i]
        gamma = self.params["gamma%d" % i]
        forward_msg, cache = batchnorm_forward(forward_msg, gamma, beta, self.bn_params[i - 1])
        batchnorm_caches.append(cache)

    if mode == 'test':
      return forward_msg

    loss, backward_msg = softmax_loss(forward_msg, y)
    for i in xrange(1, self.num_layers + 1):
      W = self.params["W%d" % i]
      loss += 0.5 * self.reg * np.sum(W * W)

    grads = {}
    for i in xrange(self.num_layers, 0, -1):
      if i != self.num_layers and self.use_batchnorm:
        cache = batchnorm_caches[i - 1]
        backward_msg, dgamma, dbeta = batchnorm_backward(backward_msg, cache)
        grads["beta%d" % i] = dbeta
        grads["gamma%d" % i] = dgamma

      if self.use_dropout:
        cache = dropout_caches[i - 1]
        backward_msg = dropout_backward(backward_msg, cache)

      cache = affine_caches[i - 1]
      if i != self.num_layers:
        backward_msg, dW, db = affine_relu_backward(backward_msg, cache)
      else:
        backward_msg, dW, db = affine_backward(backward_msg, cache)
      grads["W%d" % i] = dW + self.reg * self.params["W%d" % i]
      grads["b%d" % i] = db

    return loss, grads
