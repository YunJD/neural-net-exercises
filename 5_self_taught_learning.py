#!/usr/bin/python3

from scipy.io import loadmat
import nn
import sparse_autoencoder as sa
import mnist
import numpy as np
import gbl
import tensorflow as tf
from stochastic import *
from datetime import datetime
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('sparse_param', 0.1, 'Sparsity parameter.')
flags.DEFINE_float('sparse_pen', 3, 'Sparsity penalty.')
flags.DEFINE_float('decay', 3e-3, 'Decay parameter.')
flags.DEFINE_integer('max_steps', 20000, 'Number of training steps to run.')
flags.DEFINE_integer('hidden', 200, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 400, 'Batch size for stochastic gradient descent.')
flags.DEFINE_integer('max_steps_sm', 14000, 'Number of training steps for the softmax layer.')

def main(_):
  n, labels = mnist.read_labels('res/train-labels.idx1-ubyte')
  one_hot_labels = np.eye(10)[labels]

  n, width, height, images = mnist.read_images('res/train-images.idx3-ubyte')
  images = images.reshape([n, width * height])

  unlabeled_images = images[labels >= 5]
  labeled_images = images[labels < 5]
  labeled_labels = one_hot_labels[labels < 5]

  n, labels = mnist.read_labels('res/t10k-labels.idx1-ubyte')
  one_hot_labels = np.eye(10)[labels]

  n, width, height, images = mnist.read_images('res/t10k-images.idx3-ubyte')
  images = images.reshape([n, width * height])

  test_images = images[labels < 5]
  test_labels = one_hot_labels[labels < 5]

  with tf.Graph().as_default():
    # Autoencoder model
    x, w1, b1, a1, loss = sa.get_ae_stack(width * height, FLAGS.hidden, FLAGS.decay, FLAGS.sparse_param, FLAGS.sparse_pen)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Softmax model
    encoded_x = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    y_ = tf.placeholder(tf.float32, shape=(len(test_labels), 10))

    sm_w, sm_b = tf.Variable(tf.zeros((FLAGS.hidden, 10))), tf.Variable(tf.zeros((10,)))
    softmax = tf.nn.softmax(tf.matmul(encoded_x, sm_w) + sm_b)

    validation = 100 * tf.reduce_mean(tf.cast(
      tf.equal(tf.argmax(softmax, 1), tf.argmax(y_, 1)),
      tf.float32
    ))

    # Softmax Loss function
    sm_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax), reduction_indices=[1])) + tf.nn.l2_loss(sm_w) * 1e-4

    sm_train_op = tf.train.AdamOptimizer().minimize(sm_loss)

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    start_time = time.time()

    # Train autoencoder
    print(datetime.now(), "Training autoencoder starts...")
    batches = stochastic_batch(unlabeled_images, FLAGS.batch_size)
    for step in range(FLAGS.max_steps):
      _, loss_value = sess.run([train_op, loss], feed_dict={
        x: next(batches)
      })

      if step % 50 == 0:
        print('Step', step, 'Loss', loss_value, 'Elapsed', time.time() - start_time)

    print(datetime.now(), "Training completed! Elapsed:", time.time() - start_time, '\n')

    n_tiles = int(np.ceil(np.sqrt(FLAGS.hidden)))
    w_ = sess.run(w1).T.reshape(FLAGS.hidden, height, width)
    gbl.plot_image(
      gbl.get_tile_image(w_, n_tiles, n_tiles),
      show=False,
      filename="images/5 Self Taught Learning/filters.png",
      cmap="Greys_r"
    )

    encoded_images = sess.run(a1, feed_dict={
      x: labeled_images
    })

    encoded_test_images = sess.run(a1, feed_dict={
      x: test_images
    })

    # Train softmax classifier
    print(datetime.now(), 'Training softmax layer starts...')
    batches = stochastic_batch([encoded_images, labeled_labels], 100)

    test_feed_dict = {
      encoded_x: encoded_test_images,
      y_: test_labels
    }
    for step in range(FLAGS.max_steps_sm):
      batch = next(batches)

      loss_value = 1
      _, loss_value = sess.run([sm_train_op, sm_loss], feed_dict={
        encoded_x: batch[0],
        y: batch[1]
      })

      if step % 50 == 0:
        accuracy = sess.run(validation, feed_dict=test_feed_dict)
        print('Step', step, 'Accuracy', accuracy)

    print("Training complete! Final accuracy", sess.run(validation, feed_dict=test_feed_dict))
if __name__ == '__main__':
  tf.app.run()
