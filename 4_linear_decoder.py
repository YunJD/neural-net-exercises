#!/usr/bin/python3

from scipy.io import loadmat
import sparse_autoencoder as sa
import nn
import numpy as np
import gbl
import tensorflow as tf
from datetime import datetime
from stochastic import *
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('sparse_param', 0.035, 'Sparsity parameter.')
flags.DEFINE_float('sparse_pen', 5, 'Sparsity penalty.')
flags.DEFINE_float('decay', 3e-3, 'Decay parameter.')
flags.DEFINE_float('zca_reg', 0.1, 'Regularization parameter for ZCA whitening.')
flags.DEFINE_integer('max_steps', 25000, 'Number of training steps to run.')
flags.DEFINE_integer('hidden', 400, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 200, 'Batch size for stochastic gradient descent.')

def main(_):
  # Data loading and processing
  # Flip axes because UFLDL loads in column vectors
  patches = loadmat('res/stlSampledPatches.mat', mat_dtype=True)['patches']\
    .reshape([3, 8, 8, 100000])\
    .swapaxes(0, -1).swapaxes(1, 2)

  gbl.plot_image(
    gbl.get_tile_image(patches[0:900], 30, 30, normalize=False),
    show=False,
    filename='images/4 Linear Decoder/stl patches.png',
    interpolation='NEAREST'
  )

  patches = patches.reshape([100000, 192])
  patches -= patches.mean(0)

  # ZCA whitening
  u, U = np.linalg.eig(np.dot(patches.T, patches) / patches.shape[0])
  zca_whiten = np.dot(U / np.sqrt(u + FLAGS.zca_reg), U.T)
  zca_patches = np.dot(patches, zca_whiten)

  gbl.plot_image(
    gbl.get_tile_image(zca_patches[0:900].reshape([900, 8, 8, 3]), 30, 30),
    show=False,
    filename='images/4 Linear Decoder/stl zca whitened patches.png',
    interpolation='NEAREST'
  )

  with tf.Graph().as_default():
    # Tensorflow model
    x = tf.placeholder(tf.float32, shape=(None, 192))
    w1, b1, z1, a1 = nn.feed_forward(x, FLAGS.hidden)
    w2, b2, z2, a2 = nn.feed_forward(a1, 192, True)

    # Autoencoder loss function
    loss = nn.square_error_loss(a2, x)\
      + nn.weight_decay(FLAGS.decay, w1, w2)\
      + nn.sparsity(a1, FLAGS.sparse_param, FLAGS.sparse_pen)

    # Tensorflow Training
    train_op = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    print(datetime.now(), "Begin training...")
    start_time = time.time()

    batches = stochastic_batch(zca_patches, FLAGS.batch_size)
    for step in range(FLAGS.max_steps):
      _, loss_value = sess.run([train_op, loss], feed_dict={
        x: next(batches)
      })

      if step % 50 == 0:
        print('Steps', step, 'Loss', loss_value, 'Elapsed', time.time() - start_time)

        w_ = sess.run(w1).T.reshape(FLAGS.hidden, 8, 8, 3)
        gbl.plot_image(
          gbl.get_tile_image(w_, 20, 20),
          show=False,
          filename='images/4 Linear Decoder/anim/stl filters-{0}.png'.format(step),
          interpolation='NEAREST'
        )

    print(datetime.now(), "Training complete! Elapsed:", time.time() - start_time)

    w_ = sess.run(w1).T.reshape(FLAGS.hidden, 8, 8, 3)
    gbl.plot_image(
      gbl.get_tile_image(w_, 20, 20),
      show=False,
      filename='images/4 Linear Decoder/stl filters.png',
      interpolation='NEAREST'
    )

    saver.save(sess, 'data/linear_decoder/state')

if __name__ == '__main__':
  tf.app.run()
