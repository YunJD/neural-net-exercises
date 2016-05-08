import numpy as np
import struct

__all__ = ['MNISTImages', 'MNISTLabels', 'read_images', 'read_labels']

def read_labels(filepath, one_hot=False):
  with open(filepath, 'rb') as f:
    magic, n = struct.unpack(">II", f.read(8))
    if magic != 2049:
      raise Exception("Unexpected magic number {0}, expected 2049".format(magic))

    if one_hot:
      labels = np.eye(10, dtype=np.float32)[np.fromfile(f, np.uint8)]
    else:
      labels = np.fromfile(f, np.uint8).astype(np.float32)

  return n, labels

def read_images(filepath):
  with open(filepath, 'rb') as f:
    magic, n, w, h = struct.unpack(">IIII", f.read(16))
    if magic != 2051: 
      raise Exception("Unexpected magic number {0}, expected 2051".format(magic)) 

    images = np.fromfile(f, np.uint8).astype(np.float32) / 255.

  return n, w, h, images

class MNISTImages:
  def __init__(self, filepath, flat=False):
    self.flat = flat
    self.n, self.width, self.height, self.images = read_images(filepath)
    self.validation = None

    if flat:
      self.images = self.images.reshape([self.n, self.width * self.height])
    else:
      self.images = self.images.reshape([self.n, self.width, self.height])

  def split_validation_set(self, n = 5000):
    if self.validation is None:
      self.images, self.validation = self.images[:self.n - n], self.images[self.n - n:]

    return self.images, self.validation

  def get_images(self):
    return self.images

  def get_validation(self):
    return self.validation

class MNISTLabels:
  def __init__(self, filepath, one_hot=False):
    self.one_hot = one_hot
    self.n, self.labels = read_labels(filepath, one_hot)
    self.validation = None

  def split_validation_set(self, n = 5000):
    if self.validation is None:
      self.labels, self.validation = self.labels[:self.n - n], self.labels[self.n - n:]

    return self.labels, self.validation

  def get_labels(self):
    return self.labels

  def get_validation(self):
    return self.validation
