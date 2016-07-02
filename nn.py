import numpy as np
import tensorflow as tf

def feed_forward(prev_layer, n, is_linear=False, dev=None):
  '''Generates everything needed for a single feed forward step.

  Args:
    prev_layer (tensorflow.Tensor): Previous activation/input layer.
    n (integer): Number of nodes.
    is_linear (boolean = False): Activation is the same as z.
    dev (float = np.sqrt(6. / (n_prev_layer + n + 1.)): Min/max range for randomized weights.

  Returns:
    (tuple): Tuple containing:
      w (tensorflow.Tensor): Weights.
      b (tensorflow.Tensor): Biases.
      z (tensorflow.Tensor): prev_layer * w + b.
      a (tensorflow.Tensor): Activation for the feed forward pass.
  '''
  n_prev = int(prev_layer.get_shape()[-1])

  dev = dev or np.sqrt(6. / (n_prev + n + 1))

  w = tf.Variable(tf.random_uniform((n_prev, n), -dev, dev))
  b = tf.Variable(tf.zeros((n,)))
  z = tf.matmul(prev_layer, w) + b
  a = z if is_linear else tf.sigmoid(z)

  return w, b, z, a

def square_error_loss(activations, y):
  '''Square error loss.

  Args:
    activations (tensorflow.Tensor): Activations for output layer.
    y (tensorflow.Tensor): Expected results given training example.

  Returns:
    (tensorflow.Tensor): Square error loss.
  '''

  # Since activations are vectors of shape [n_batches, n_classes], must sum 
  # using reduction_indices=[1] first to produce a vector of shape [n_batches]
  # otherwise reduce_mean will divide by n_batches * n_classes which is wrong.
  return tf.reduce_mean(
    tf.reduce_sum(tf.squared_difference(activations, y), reduction_indices=[1])
  ) / 2.

def cross_entropy_loss(activations, y):
  '''Cross entropy loss.

  Args:
    activations (tensorflow.Tensor): Activations for output layer.
    y (tensorflow.Tensor): Expected results given training example.

  Returns:
    (tensorflow.Tensor): Cross entropy loss.
  '''

  return tf.reduce_mean(
    tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(activations, y), reduction_indices=[1])
  )

def weight_decay(penalty, *weights):
  if len(weights) == 0:
    return 0

  weight_decays = tf.nn.l2_loss(weights[0])
  for i in range(1, len(weights)):
    weight_decays += tf.nn.l2_loss(weights[i])

  return penalty * weight_decays

def sparsity(a, p, b):
  if p == 0 or b == 0:
    return 0

  p_hat = tf.reduce_mean(a, 0)
  kl = tf.reduce_sum(p * tf.log(p / p_hat) + (1. - p) * tf.log((1. - p) / (1. - p_hat)))

  return b * kl
