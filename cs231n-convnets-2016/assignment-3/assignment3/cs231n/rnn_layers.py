import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  a = prev_h.dot(Wh) + x.dot(Wx) + b
  next_h = np.tanh(a)
  cache = (x, prev_h, Wx, Wh, a, next_h)
  return next_h, cache


def rnn_step_backward(d_next_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - d_next_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - d_x: Gradients of input data, of shape (N, D)
  - d_prev_h: Gradients of previous hidden state, of shape (N, H)
  - d_Wx: Gradients of input-to-hidden weights, of shape (D, H)
  - d_Wh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - d_b: Gradients of bias vector, of shape (H,)
  """
  x, prev_h, Wx, Wh, a, next_h = cache

  d_a = (1 - next_h * next_h) * d_next_h

  d_prev_h = d_a.dot(Wh.T)
  d_Wh = prev_h.T.dot(d_a)

  d_x = d_a.dot(Wx.T)
  d_Wx = x.T.dot(d_a)

  d_b = np.sum(d_a, axis=0)

  return d_x, d_prev_h, d_Wx, d_Wh, d_b


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((N, T, H))
  h_prev = h0
  cache = []
  for t in xrange(T):
    h_next, step_cache = rnn_step_forward(x[:,t,:], h_prev, Wx, Wh, b)
    h[:,t,:] = h_next
    h_prev = h_next
    cache.append(step_cache)
  return h, cache


def rnn_backward(d_h, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - d_h: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - d_x: Gradient of inputs, of shape (N, T, D)
  - d_h0: Gradient of initial hidden state, of shape (N, H)
  - d_Wx: Gradient of input-to-hidden weights, of shape (D, H)
  - d_Wh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - d_b: Gradient of biases, of shape (H,)
  """
  N, T, H = d_h.shape
  N, D = cache[0][0].shape

  d_x = np.zeros((N, T, D))
  d_h0 = np.zeros((N, H))
  d_Wx = np.zeros((D, H))
  d_Wh = np.zeros((H, H))
  d_b = np.zeros((H, ))

  d_h_next = np.zeros((N, H))
  for t in reversed(xrange(T)):
    d_x_step, d_prev_h_step, d_Wx_step, d_Wh_step, d_b_step = rnn_step_backward(d_h[:,t,:] + d_h_next, cache[t])
    d_h_next = d_prev_h_step

    d_x[:,t,:] = d_x_step
    d_h0 = d_prev_h_step
    d_Wx += d_Wx_step
    d_Wh += d_Wh_step
    d_b += d_b_step

  return d_x, d_h0, d_Wx, d_Wh, d_b


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on mini-batches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x must be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  # The long version:
  #
  # for n in xrange(N):
  #   for t in xrange(T):
  #     v = x[n, t]
  #     for d in xrange(D):
  #       out[n, t, d] = W[v, d]
  #
  # <alternative version>
  #
  # for n in xrange(N):
  #   for t in xrange(T):
  #     v = x[n, t]
  #     out[n, t, :] = W[v, :]
  #
  out = W[x,:]
  cache = (x, W)
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  x, W = cache
  dW = np.zeros_like(W)

  # The long version:
  #
  # for n in xrange(N):
  #   for t in xrange(T):
  #     v = x[n, t]
  #     for d in xrange(D):
  #       dW[v, d] += dout[n, t, d]
  #
  # <alternative version>
  #
  # for n in xrange(N):
  #   for t in xrange(T):
  #     v = x[n, t]
  #     dW[v, :] += dout[n, t, :]
  #
  np.add.at(dW, x, dout)
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  N, H = prev_h.shape
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  i = sigmoid(a[:,   :  H])
  f = sigmoid(a[:,  H:2*H])
  o = sigmoid(a[:,2*H:3*H])
  g = np.tanh(a[:,3*H:   ])
  next_c = f * prev_c + i * g
  z = np.tanh(next_c)
  next_h = o * z
  cache = x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, z, next_h
  return next_h, next_c, cache


def lstm_step_backward(d_next_h, d_next_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, z, next_h = cache

  d_z = o * d_next_h
  d_o = z * d_next_h
  d_next_c += (1 - z * z) * d_z

  d_f = d_next_c * prev_c
  d_prev_c = d_next_c * f
  d_i = d_next_c * g
  d_g = d_next_c * i

  d_a_g = (1 - g * g) * d_g
  d_a_o = o * (1 - o) * d_o
  d_a_f = f * (1 - f) * d_f
  d_a_i = i * (1 - i) * d_i
  d_a = np.concatenate((d_a_i, d_a_f, d_a_o, d_a_g), axis=1)

  d_prev_h = d_a.dot(Wh.T)
  d_Wh = prev_h.T.dot(d_a)

  d_x = d_a.dot(Wx.T)
  d_Wx = x.T.dot(d_a)

  d_b = np.sum(d_a, axis=0)

  return d_x, d_prev_h, d_prev_c, d_Wx, d_Wh, d_b


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((N, T, H))
  cache = []

  h_prev = h0
  c_prev = np.zeros_like(h0)
  for t in xrange(T):
    h_next, c_next, step_cache = lstm_step_forward(x[:, t, :], h_prev, c_prev, Wx, Wh, b)
    h[:,t,:] = h_next
    h_prev = h_next
    c_prev = c_next
    cache.append(step_cache)

  return h, cache


def lstm_backward(d_h, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  N, T, H = d_h.shape
  N, D = cache[0][0].shape

  d_x = np.zeros((N, T, D))
  d_h0 = np.zeros((N, H))
  d_Wx = np.zeros((D, 4*H))
  d_Wh = np.zeros((H, 4*H))
  d_b = np.zeros((4*H, ))

  d_h_next_t = np.zeros((N, H))
  d_c_next_t = np.zeros((N, H))
  for t in reversed(xrange(T)):
    d_x_t, d_h_prev_t, d_c_prev_t, d_Wx_t, d_Wh_t, d_b_t = lstm_step_backward(d_h_next_t + d_h[:,t,:], d_c_next_t, cache[t])
    d_c_next_t = d_c_prev_t
    d_h_next_t = d_h_prev_t

    d_x[:,t,:] = d_x_t
    d_h0 = d_h_prev_t
    d_Wx += d_Wx_t
    d_Wh += d_Wh_t
    d_b += d_b_t

  return d_x, d_h0, d_Wx, d_Wh, d_b


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

