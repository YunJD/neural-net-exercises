#!/usr/bin/python3

from scipy.io import loadmat
from sparse_autoencoder_scipy import SimpleSAE
import numpy as np
import gbl

################################################################################
# Parameters
################################################################################
SPARSITY_PARAM = 0.035
DECAY = 3e-3
BETA = 5

EPSILON = 0.1

INPUTS = 192
HIDDEN = 400

nn = SimpleSAE([192, HIDDEN, 192], DECAY, SPARSITY_PARAM, BETA, True)

################################################################################
# Data loading and processing
################################################################################
#Flip axes because UFLDL loads in column vectors
patches = loadmat('res/stlSampledPatches.mat', mat_dtype=True)['patches']\
  .reshape([3, 8, 8, 100000])\
  .swapaxes(0, -1).swapaxes(1, 2)

gbl.plot_image(
  gbl.get_tile_image(patches[0:400], 20, 20, normalize=False),
  filename='images/4 linear decoder/stl patches.png',
  interpolation='NEAREST'
)

patches = patches.reshape([100000, 192])
patches -= patches.mean(0)

u, U = np.linalg.eig(np.dot(patches.T, patches) / patches.shape[0])
zca_whiten = np.dot(U / np.sqrt(u + EPSILON), U.T)
zca_patches = np.dot(patches, zca_whiten)

gbl.plot_image(
  gbl.get_tile_image(zca_patches[0:400].reshape([400, 8, 8, 3]), 20, 20),
  filename='images/4 linear decoder/stl zca whitened patches.png',
  interpolation='NEAREST'
)
################################################################################
# Super simplistic upgrade to the simple sparse autoencoder
################################################################################
nn.optimize(zca_patches, zca_patches, 50)

w_ = nn.w[0].T.reshape(HIDDEN, 8, 8, 3)
gbl.plot_image(
  gbl.get_tile_image(w_, 20, 20),
  filename='images/4 linear decoder/stl filters.png',
  interpolation='NEAREST'
)
