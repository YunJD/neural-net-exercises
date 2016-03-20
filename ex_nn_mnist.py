#!/usr/bin/python3

#Applies a 3-layer neural net to the MNIST handwritten digits

from mnist import *
from nn import SimpleNN
import numpy as np
from sys import stdout as cout

training_labels = MNISTLabels("res/train-labels.idx1-ubyte", True)
training_labels.split_validation_set()

training_images = MNISTImages("res/train-images.idx3-ubyte", True)
training_images.split_validation_set()

nn = SimpleNN([training_images.width * training_images.height, 80, 10], 7.5e-5)
nn.learn(
  50000,
  100,
  0.4,
  training_images.get_images(),
  training_labels.get_labels(),
  True
)

print('Validating')

testing_labels = MNISTLabels('res/t10k-labels.idx1-ubyte', True)
testing_images = MNISTImages('res/t10k-images.idx3-ubyte', True)

r = nn.feed_forward(training_images.get_validation())
validation = r.argmax(1) == training_labels.get_validation().argmax(1)

r = nn.feed_forward(testing_images.get_images())
test = r.argmax(1) == testing_labels.get_labels().argmax(1)

print('Validation accuracy: {0}%'.format(100. * np.sum(validation) / len(training_labels.get_validation())))
print('Testing accuracy: {0}%'.format(100 * np.sum(test) / len(testing_labels.get_labels())))
