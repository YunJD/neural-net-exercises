#!/usr/bin/python3

#Applies a 3-layer sparse autoencoder to a randomized set of image patches

from scipy.io import loadmat
from sparse_autoencoder_scipy import SimpleSAE
import numpy as np
from sys import stdout as cout
import gbl
import matplotlib.pyplot as plt

p_w = 8
p_h = 8
imgs = loadmat('res/IMAGES.mat', mat_dtype=True)['IMAGES'].swapaxes(2,1).swapaxes(1,0)
patches = gbl.sample_random_patches(imgs, 10000, p_w, p_h)

mean = patches.mean()
std = 3 * patches.std()
patches = 0.4 * (np.maximum(np.minimum(patches - mean, std), -std) / std + 1) + 0.1
patches = patches.reshape([10000, p_w * p_h])

s2 = 5
nn = SimpleSAE(
  layers=[patches.shape[1], s2 * s2, patches.shape[1]],
  decay=1e-4,
  sparsity=0.01,
  sparsity_penalty=3
)
nn.optimize(patches, patches, 400)

#TODO: create an unpatchify function
#Transpose because column matrices are used for the weights
hidden_w = nn.w[0].T
#Matrix notation reverses height vs width

figure, axes = plt.subplots(nrows = s2, ncols = s2)
index = 0

for axis in axes.flat:
  image = axis.imshow(nn.w[0].T[index].reshape(p_w, p_h),
    cmap = plt.cm.gray, interpolation = 'nearest'
  )
  axis.set_frame_on(False)
  axis.set_axis_off()
  index += 1

plt.show()
