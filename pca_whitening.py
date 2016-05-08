import tensorflow as tf

def pca_whitening(patch_size):
  x = tf.placeholder(tf.float32, [None, patch_size])
