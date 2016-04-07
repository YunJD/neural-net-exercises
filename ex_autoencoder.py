#!/usr/bin/python3

#Applies a 3-layer sparse autoencoder to a randomized set of image patches

from sparse import *
from nn import SimpleNN, SimpleSoftmaxNN
import numpy as np
from sys import stdout as cout
import matplotlib.pyplot as plt

p_w = 8
p_h = 8
imgs, patches = sample_patches(p_w, p_h, True)

s2 = 5
nn = SimpleNN([patches.shape[-1], s2 * s2, patches.shape[-1]], 1e-4)
#Simply use plain old gradient descent for this exercise, not worth trying to implement L-BFGS
try:
  nn.optimize(
    300000,
    100,
    0.02,
    patches,
    patches,
    True,
    0.007,
    6
  )
except:
  pass

hidden_vis = nn.w[0].reshape([s2 * s2,p_w,p_h])
#hidden_vis = hidden_vis / (np.sqrt(np.sum(np.power(hidden_vis2, 2), axis=(1,2), keepdims=True)))
weights_all = None

a_avg = np.average(np.dot(patches, nn.w[0]), axis=0)
#hidden_vis2 = np.power(hidden_vis, 2)
import pdb; pdb.set_trace()
for y in range(0, s2):
    patches_row = np.array(hidden_vis[y * s2])
    for x in range(1, s2):
        patches_row = np.append(patches_row, hidden_vis[y * s2 + x], axis=1)
    if weights_all is None:
        weights_all = patches_row
    else:
        weights_all = np.append(weights_all, patches_row, axis=0)

plt.imshow(weights_all, cmap='Greys_r')
plt.show()
