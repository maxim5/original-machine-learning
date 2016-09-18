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
