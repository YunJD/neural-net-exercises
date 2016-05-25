#!/usr/bin/python3

from mnist import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import gbl

plt.axis('off')

patch_size = 50

imgs = loadmat('res/IMAGES_RAW.mat', mat_dtype=True)['IMAGESr']
#We use row-vectors rather than column-vectors, which maps directly to matplotlib
imgs = imgs.swapaxes(0, 2).swapaxes(1, 2)

#Show the images
gbl.plot_image(
  gbl.get_tile_image(imgs, 3, 4),
  filename='images/3 PCA/1 - Raw Natural Images.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)

patches = gbl.sample_random_patches(imgs, 10000, patch_size, patch_size)

patches = patches.reshape([10000, patch_size * patch_size])
patches = patches - patches.mean(1, keepdims=True)

random_patches = np.random.randint(0, patches.shape[0], 100)
#Show 25 random patches

gbl.plot_image(
  gbl.get_tile_image(patches.reshape([10000, patch_size, patch_size])[random_patches], 10, 10),
  filename='images/3 PCA/2 - Randomly Sampled Patches.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)

#Compute the cov matrix and xrot.
cov = np.dot(patches.T, patches) / 10000
u, U = np.linalg.eig(cov)
rot = np.dot(patches, U)

#Save the cov matrix of xrot
plt.imshow((np.dot(rot.T, rot) / 10000), interpolation='NEAREST')
plt.savefig('images/3 PCA/3 - Covariance Matrix.svg', bbox_inches='tight')
#This sorts from smallest to largest
idx = u.argsort()

#Calculate the cumulative sums, getting the proportions at the end
eig_sum = [0]
for i in range(len(idx)):
  eig_sum.append(eig_sum[-1] + u[idx[i]])

eig_sum = np.array(eig_sum)
eig_sum /= eig_sum[-1]

#Note that we're going backwards compared to the notes.  So, to keep 90% or 99% variance, we find k elements that represent up to 10% or 1% of the cumulative proportions, and remove those indices

#This function hopefully implements binary search.  Why wouldn't it?
idx_90, idx_99 = np.searchsorted(eig_sum, [0.1, 0.01])

pca_90 = rot.copy()
pca_99 = rot.copy()
pca_90[:,idx[0:idx_90 + 1]] = 0.
pca_99[:,idx[0:idx_99 + 1]] = 0.
pca_90 = np.dot(pca_90, U.T)
pca_99 = np.dot(pca_99, U.T)

rot_white = rot / np.sqrt(u)
rot_white_reg = rot / (np.sqrt(u + 0.1))

plt.imshow(np.dot(rot_white.T, rot_white) / 10000, interpolation='NEAREST')
plt.savefig('images/3 PCA/4 - Covariance Matrix Whitened.svg', bbox_inches='tight')
plt.imshow(np.dot(rot_white_reg.T, rot_white_reg) / 10000, interpolation='NEAREST')
plt.savefig('images/3 PCA/5 - Covariance Matrix Whitened Regularized.svg', bbox_inches='tight')

gbl.plot_image(
  gbl.get_tile_image(pca_90.reshape([10000, patch_size, patch_size])[random_patches], 10, 10),
  filename='images/3 PCA/6 - PCA 90 Variance.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)

gbl.plot_image(
  gbl.get_tile_image(pca_99.reshape([10000, patch_size, patch_size])[random_patches], 10, 10),
  filename='images/3 PCA/7 - PCA 99 Variance.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)

rot_white_reg_pca_90 = rot_white_reg.copy()
rot_white_reg_pca_99 = rot_white_reg.copy()
rot_white_reg_pca_90[:,idx[0:idx_90 + 1]] = 0.
rot_white_reg_pca_99[:,idx[0:idx_99 + 1]] = 0.
rot_white_reg_zca_90 = np.dot(rot_white_reg_pca_90, U.T)
rot_white_reg_zca_99 = np.dot(rot_white_reg_pca_99, U.T)

gbl.plot_image(
  gbl.get_tile_image(rot_white_reg_zca_90.reshape([10000, patch_size, patch_size])[random_patches], 10, 10),
  filename='images/3 PCA/8 - ZCA 90 Variance.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)

gbl.plot_image(
  gbl.get_tile_image(rot_white_reg_zca_99.reshape([10000, patch_size, patch_size])[random_patches], 10, 10),
  filename='images/3 PCA/9 - ZCA 99 Variance.png',
  cmap=cm.gray,
  interpolation='NEAREST'
)
