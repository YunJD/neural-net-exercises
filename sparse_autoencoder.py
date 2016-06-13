import numpy as np
import tensorflow as tf

def autoencoder(in_placeholder, n_layers, is_linear=True):
  in_shape = in_placeholder.get_shape()
  n_in = in_shape[-1]

  n_layers = [int(n_in)] + n_layers + [int(n_in)]

  layers = [in_placeholder]
  weight_layers = []

  rng_range = 6. / np.sqrt(np.sum(n_layers[1:]) + 1.)

  for i in range(len(n_layers) - 1):
    with tf.name_scope('layer' + str(i)):
      weights = tf.Variable(
        tf.random_uniform(
          [n_layers[i], n_layers[i + 1]],
          -rng_range, rng_range
        ),
        name='weights'
      )

      # Needed for optional weight decay
      weight_layers.append(weights)

      biases = tf.Variable(tf.zeros((n_layers[i + 1],)), name='biases')

      if i == len(n_layers) - 2 and is_linear:
        layers.append(tf.matmul(layers[-1], weights) + biases)
      else:
        layers.append(
          tf.nn.sigmoid(tf.matmul(layers[-1], weights) + biases)
        )

  return layers, weight_layers

def loss(layers, weights, in_placeholder, decay = 1e-4, sparsity_param=0, sparsity_penalty=0):
  inv_m = 1. / float(int(in_placeholder.get_shape()[0]))
  l2_loss = tf.nn.l2_loss(layers[-1] - in_placeholder) * inv_m

  weight_decays = tf.nn.l2_loss(weights[0])
  for i in range(1, len(weights)):
    weight_decays += tf.nn.l2_loss(weights[i])

  reg_loss = decay * weight_decays

  loss = l2_loss + reg_loss

  if sparsity_param and sparsity_penalty:
    for i in range(1, len(layers) - 1):
      p_hat = tf.reduce_mean(layers[i], 0)
      p = sparsity_param
      kl = tf.reduce_sum(p * tf.log(p / p_hat) + (1. - p) * tf.log((1. - p) / (1. - p_hat)))
      sparsity_term = sparsity_penalty * kl

    loss += sparsity_term
    return loss, reg_loss, sparsity_term

  return loss, reg_loss

def training(loss):
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss)

  return train_op
