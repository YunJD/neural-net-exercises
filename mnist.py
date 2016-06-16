import numpy as np
import struct

__all__ = ['read_images', 'read_labels']

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
