#!/usr/bin/python3

#Applies a 3-layer neural net to the MNIST handwritten digits

from nn_scipy import SimpleSoftmaxNN
import mnist
import numpy as np
from sys import stdout as cout

n_validation = 5000

n, labels = mnist.read_labels('res/train-labels.idx1-ubyte', True)
n, width, height, images = mnist.read_images('res/train-images.idx3-ubyte')
images = images.reshape([n, width * height])

nn = SimpleSoftmaxNN([width * height, 140, 10], 2e-5)
try:
  nn.optimize(
    50000,
    100,
    0.3,
    images[n_validation:],
    labels[n_validation:],
    True
  )
except:
  pass

print('Validating')

n, test_labels = mnist.read_labels('res/t10k-labels.idx1-ubyte', True)
n, width, height, test_images = mnist.read_images('res/t10k-images.idx3-ubyte')
test_images = test_images.reshape([n, width * height])

r = nn.feed_forward(images[:n_validation])
validation = r.argmax(1) == labels[:n_validation].argmax(1)

r = nn.feed_forward(test_images)
test = r.argmax(1) == test_labels.argmax(1)

print('Validation accuracy: {0}%'.format(100. * np.sum(validation) / n_validation))
print('Testing accuracy: {0}%'.format(100 * np.sum(test) / len(test_labels)))
