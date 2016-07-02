#!/usr/bin/python3

# This assignment is a bit fishy since we don't really get alot of unlabelled 
# data. In fact basic backprop is able to produce about the same accuracy
# depending on hyperparameters.

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
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 200, 'Number of hidden units.')
flags.DEFINE_integer('hidden2', 150, 'Number of hidden units.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for stochastic gradient descent.')
flags.DEFINE_integer('max_steps1', 60000, 'Number of training steps to run for encoder layers.')
flags.DEFINE_integer('max_steps2', 40000, 'Number of training steps to run for encoder layers.')
flags.DEFINE_integer('max_steps_sm', 10000, 'Number of training steps to run for encoder layers.')

def main(_):
  n, train_labels = mnist.read_labels('res/train-labels.idx1-ubyte')
  train_labels = np.eye(10)[train_labels]

  n, width, height, train_images = mnist.read_images('res/train-images.idx3-ubyte')
  train_images = train_images.reshape([n, width * height])

  #u, U = np.linalg.eig(np.dot(train_images.T, train_images) / train_images.shape[0])
  #zca_whiten = np.dot(U / np.sqrt(u + 0.1), U.T)
  #train_images = np.dot(train_images, zca_whiten)

  n, test_labels = mnist.read_labels('res/t10k-labels.idx1-ubyte')
  test_labels = np.eye(10)[test_labels]

  n, width, height, test_images = mnist.read_images('res/t10k-images.idx3-ubyte')
  test_images = test_images.reshape([n, width * height])
  #test_images = np.dot(test_images, zca_whiten)

  # For visualization of hidden layers
  n_tiles = int(np.ceil(np.sqrt(FLAGS.hidden1)))
  n_tiles2 = int(np.ceil(np.sqrt(FLAGS.hidden2)))

  with tf.Graph().as_default():
    opt = tf.train.AdamOptimizer()
    # Autoencoder 1
    x1, w1, b1, a1, loss1 = sa.get_ae_stack(
      width * height, FLAGS.hidden1, 3e-3, 0.1, 3
    )
    train1 = opt.minimize(loss1)

    # Autoencoder 2
    x2, w2, b2, a2, loss2 = sa.get_ae_stack(
      FLAGS.hidden1, FLAGS.hidden2, 1e-4, 0.15, 4
    )
    train2 = opt.minimize(loss2)

    vis2 = tf.matmul(tf.maximum(tf.transpose(w2), 0), tf.transpose(w1))

    # Softmax
    sm_x = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden2))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    y_ = tf.placeholder(tf.float32, shape=(len(test_labels), 10))

    sm_w, sm_b = tf.Variable(tf.random_uniform((FLAGS.hidden2, 10), -1e-3, 1e-3)), tf.Variable(tf.zeros((10,)))
    softmax = tf.nn.softmax(tf.matmul(sm_x, sm_w) + sm_b)

    sm_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax), reduction_indices=[1]))
    sm_train = opt.minimize(sm_loss)

    # Fully connected model
    full = sa.fully_connect_sigmoid_softmax(a1, (w2, b2), (sm_w, sm_b))
    full_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(full), reduction_indices=[1]))
    fine_tune_train = tf.train.GradientDescentOptimizer(0.15).minimize(full_loss)

    validation = 100 * tf.reduce_mean(tf.cast(
      tf.equal(tf.argmax(full, 1), tf.argmax(y_, 1)), tf.float32
    ))

    test_feed_dict = {
      x1: test_images,
      y_: test_labels
    }

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    start_time = time.time()

    restored = os.path.isfile('data/sae/state')
    if restored:
      saver.restore(sess, 'data/sae/state')
    else:
      # Train autoencoder 1
      print(datetime.now(), "Training autoencoder 1 starts...")
      batches = stochastic_batch(train_images, FLAGS.batch_size)
      for step in range(FLAGS.max_steps1):
        _, loss_value = sess.run([train1, loss1], feed_dict={
          x1: next(batches)
        })

        if step % 50 == 0:
          print('Step', step, 'Loss', loss_value, 'Elapsed', time.time() - start_time)

      print(datetime.now(), "Training completed! Elapsed:", time.time() - start_time, '\n')

      w_ = sess.run(w1).T.reshape(FLAGS.hidden1, height, width)
      gbl.plot_image(
        gbl.get_tile_image(w_, n_tiles, n_tiles),
        show=False,
        filename="vis/6 Stacked Autoencoder/filters 1.png",
        cmap="Greys_r"
      )

    encoded_images = sess.run(a1, feed_dict={
      x1: train_images
    })

    # Train autoencoder 2

    if not restored:
      print(datetime.now(), "Training autoencoder 2 starts...")
      batches = stochastic_batch(encoded_images, FLAGS.batch_size)
      for step in range(FLAGS.max_steps2):
        _, loss_value = sess.run([train2, loss2], feed_dict={
          x2: next(batches)
        })

        if step % 50 == 0:
          print('Step', step, 'Loss', loss_value, 'Elapsed', time.time() - start_time)

      gbl.plot_image(
        gbl.get_tile_image(sess.run(vis2).reshape(FLAGS.hidden2, height, width), n_tiles2, n_tiles2),
        show=False,
        filename="vis/6 Stacked Autoencoder/filters 2.png",
        cmap="Greys_r"
      )

      print(datetime.now(), "Training completed! Elapsed:", time.time() - start_time, '\n')

    saver.save(sess, 'data/sae/state')

    encoded_images = sess.run(a2, feed_dict={
      x2: encoded_images
    })

    # Train softmax classifier
    print(datetime.now(), 'Training softmax layer starts...')
    batches = stochastic_batch((encoded_images, train_labels), 200)

    for step in range(FLAGS.max_steps_sm):
      batch = next(batches)

      _, loss_value = sess.run([sm_train, sm_loss], feed_dict={
        sm_x: batch[0],
        y: batch[1]
      })

      if step % 50 == 0:
        accuracy = sess.run(validation, feed_dict=test_feed_dict)
        print('Step', step, 'Accuracy', accuracy, 'Loss', loss_value)

    print(datetime.now(), "Training completed! Elapsed:", time.time() - start_time, '\n')
    pretrain_acc = sess.run(validation, feed_dict=test_feed_dict)

    # Fine tuning
    print(datetime.now(), 'Fine-tuning starts...')
    batches = stochastic_batch((train_images, train_labels), 200)

    for step in range(30000):
      batch = next(batches)
      _, loss_value = sess.run([fine_tune_train, full_loss], feed_dict={
        x1: batch[0],
        y: batch[1]
      })

      if step % 50 == 0:
        accuracy = sess.run(validation, feed_dict=test_feed_dict)
        print('Step', step, 'Accuracy', accuracy, 'Loss', loss_value)

    print("Training complete! Pretrain accuracy", pretrain_acc, "Final accuracy", sess.run(validation, feed_dict=test_feed_dict))

    w_ = sess.run(w1).T.reshape(FLAGS.hidden1, height, width)
    gbl.plot_image(
      gbl.get_tile_image(w_, n_tiles, n_tiles),
      show=False,
      filename="vis/6 Stacked Autoencoder/fine tune filters 1.png",
      cmap="Greys_r"
    )

    gbl.plot_image(
      gbl.get_tile_image(sess.run(vis2).reshape(FLAGS.hidden2, height, width), n_tiles2, n_tiles2),
      show=False,
      filename="vis/6 Stacked Autoencoder/fine tune filters 2.png",
      cmap="Greys_r"
    )

if __name__ == '__main__':
  tf.app.run()
