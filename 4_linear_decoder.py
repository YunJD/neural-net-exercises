#!/usr/bin/python3

from scipy.io import loadmat
import sparse_autoencoder as sa
import numpy as np
import gbl
import tensorflow as tf
from datetime import datetime

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('sparse_param', 0.035, 'Sparsity parameter.')
flags.DEFINE_float('sparse_pen', 5, 'Sparsity penalty.')
flags.DEFINE_float('decay', 3e-3, 'Decay parameter.')
flags.DEFINE_float('zca_reg', 0.1, 'Regularization parameter for ZCA whitening.')
flags.DEFINE_integer('max_steps', 25000, 'Number of training steps to run.')
flags.DEFINE_integer('hidden', 400, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 1000, 'Batch size for stochastic gradient descent.')

def main(_):
  with tf.Graph().as_default():
    # Tensorflow model
    train_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 192))
    layers, weights = sa.autoencoder(train_placeholder, [FLAGS.hidden])
    weights_op = weights[0] * 1.

    # Tensorflow Loss function
    loss, reg_loss, sparsity_loss = sa.loss(layers, weights, train_placeholder, FLAGS.decay, FLAGS.sparse_param, FLAGS.sparse_pen)

    # Tensorflow Training
    train_op = sa.training(loss)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    # Data loading and processing
    # Flip axes because UFLDL loads in column vectors
    patches = loadmat('res/stlSampledPatches.mat', mat_dtype=True)['patches']\
      .reshape([3, 8, 8, 100000])\
      .swapaxes(0, -1).swapaxes(1, 2)

    gbl.plot_image(
      gbl.get_tile_image(patches[0:900], 30, 30, normalize=False),
      show=False,
      filename='images/4 linear decoder/stl patches.png',
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
      filename='images/4 linear decoder/stl zca whitened patches.png',
      interpolation='NEAREST'
    )

    rng = np.arange(zca_patches.shape[0])
    start, end = 0, FLAGS.batch_size

    print(datetime.now(), "Begin training...")
    for step in range(FLAGS.max_steps):
      _, loss_value = sess.run([train_op, loss], feed_dict={
        train_placeholder: zca_patches[rng[start:end]]
      })

      start += FLAGS.batch_size
      end += FLAGS.batch_size

      if start >= len(zca_patches):
        start, end = 0, FLAGS.batch_size
        np.random.shuffle(rng)

      if step % 50 == 0:
        print('Steps', step, 'Loss', loss_value)

        w_ = sess.run(weights_op).T.reshape(FLAGS.hidden, 8, 8, 3)
        gbl.plot_image(
          gbl.get_tile_image(w_, 20, 20),
          show=False,
          filename='images/4 linear decoder/anim/stl filters-{0}.png'.format(step),
          interpolation='NEAREST'
        )

    print(datetime.now(), "Training complete!")

    w_ = sess.run(weights_op).T.reshape(FLAGS.hidden, 8, 8, 3)
    gbl.plot_image(
      gbl.get_tile_image(w_, 20, 20),
      show=False,
      filename='images/4 linear decoder/stl filters.png',
      interpolation='NEAREST'
    )

    saver.save(sess, 'data/linear_decoder/state')

if __name__ == '__main__':
  tf.app.run()
