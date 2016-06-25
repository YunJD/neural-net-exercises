import numpy as np
import tensorflow as tf
import nn

def get_ae_stack(n_in, n_hidden, decay, p, b, is_linear=False):
  x = tf.placeholder(tf.float32, shape=(None, n_in))
  wh, bh, _, ah = nn.feed_forward(x, n_hidden)
  wo, bo, _, ao = nn.feed_forward(ah, n_in, is_linear)

  loss = nn.square_error_loss(ao, x)\
    + nn.weight_decay(decay, wh, wo)\
    + nn.sparsity(ah, p, b)

  # What's really needed by other functions
  return x, wh, bh, ah, loss

def fully_connect_sigmoid(x, *layers):
  a = x

  for i in range(0, len(layers)):
    w, b = layers[i]
    z = tf.matmul(a, w) + b
    a = tf.sigmoid(z)

  return a

def fully_connect_sigmoid_softmax(x, *layers):
  a = fully_connect_sigmoid(x, *layers[:-1])
  w, b = layers[-1]

  z = tf.matmul(a, w) + b
  a = tf.nn.softmax(z)

  return a
