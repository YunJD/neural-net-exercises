#!/usr/bin/python3

#Applies a 3-layer sparse autoencoder to a randomized set of image patches

from sparse import *
from nn import SimpleNN, SimpleSoftmaxNN
import numpy as np
from sys import stdout as cout

imgs, patches = sample_patches(8, 8, True)

nn = SimpleNN([patches.shape[-1], 25, patches.shape[-1]], 0.0001)
#Simply use plain old gradient descent for this exercise, not worth trying to implement L-BFGS
nn.optimize(
  50000,
  100,
  0.3,
  patches,
  patches,
  True,
  0.01,
  3.
)
