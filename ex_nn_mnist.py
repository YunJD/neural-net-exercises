#!/usr/bin/python3
from mnist import *
from nn import SimpleNN
import numpy as np
from sys import stdout as cout

training_labels = MNISTLabels("res/train-labels.idx1-ubyte", True)
training_labels.split_validation_set()

training_images = MNISTImages("res/train-images.idx3-ubyte", True)
training_images.split_validation_set()

nn = SimpleNN([training_images.width * training_images.height, 256, 10], 7.5e-5)

rng = np.arange(len(training_labels.get_labels()))
N_STEPS = 50000
N_BATCH = 100
start, end = 0, N_BATCH

try:
  for i in range(N_STEPS):
    if start >= len(rng):
      start, end = 0, N_BATCH
      if N_BATCH < len(rng):
        np.random.shuffle(rng)

    cost = nn.back_propagation(
      training_images.get_images()[rng[start:end]], training_labels.get_labels()[rng[start:end]],
      0.4
    )

    if i % 50 == 0:
      print(str(round(cost, 2)) + ' ' * 5)
      cout.write('Training...{0}%{1}\r'.format(100. * i / N_STEPS, ' ' * 5))

    start, end = start + N_BATCH, end + N_BATCH
except:
  pass

print('Training...done' + ' ' * 5)
print('Validating')

testing_labels = MNISTLabels('res/t10k-labels.idx1-ubyte', True)
testing_images = MNISTImages('res/t10k-images.idx3-ubyte', True)

r = nn.feed_forward(training_images.get_validation())
validation = r.argmax(1) == training_labels.get_validation().argmax(1)

r = nn.feed_forward(testing_images.get_images())
test = r.argmax(1) == testing_labels.get_labels().argmax(1)



print('Validation accuracy: {0}%'.format(100. * np.sum(validation) / len(training_labels.get_validation())))
print('Testing accuracy: {0}%'.format(100 * np.sum(test) / len(testing_labels.get_labels())))
