#!/usr/bin/python3

#Applies a 3-layer sparse autoencoder to a randomized set of image patches

from sparse import *
from sparse_autoencoder import SimpleSAE
import numpy as np
from sys import stdout as cout
import matplotlib.pyplot as plt

p_w = 8
p_h = 8
imgs, patches = sample_patches(p_w, p_h, True)
img = imgs[0]

s2 = 5
nn = SimpleSAE(
  layers=[patches.shape[1], s2 * s2, patches.shape[1]],
  decay=1e-4,
  sparsity=0.12,
  sparsity_penalty=10
)
nn.optimize(patches, patches, 600)

img_patches = []
for y in range(0, img.shape[0], p_h):
  for x in range(0, img.shape[1], p_w):
    img_patches.append(img[y:y + p_h, x:x + p_w])

img_patches = np.array(img_patches)
reconstruct = nn.feed_forward(img_patches.reshape(img_patches.shape[0], p_w * p_h))
reconstruct = reconstruct.reshape([img_patches.shape[0], p_h, p_w])

#TODO: create a patchify function
reconstruct_img = None
for y in range(0, int(512 / p_h)):
  reconstruct_row = reconstruct[y * (512 / p_w)]
  for x in range(1, int(512 / p_w)):
    reconstruct_row = np.append(reconstruct_row, reconstruct[y * 512 / p_w + x], axis=1)

  if reconstruct_img is None:
    reconstruct_img = reconstruct_row
  else:
    reconstruct_img = np.append(reconstruct_img, reconstruct_row, axis=0)

comp_img = np.append(img, reconstruct_img, axis=-1)
plt.imshow(comp_img, cmap='Greys_r', interpolation='nearest')
plt.show()

#TODO: create a unpatchify function
#Transpose because column matrices are used for the weights
hidden_w = nn.w[0].T
#Matrix notation reverses height vs width
min_w = np.min(hidden_w, axis=1, keepdims=True)
max_w = np.max(hidden_w, axis=1, keepdims=True)

figure, axes = plt.subplots(nrows = s2, ncols = s2)
index = 0
                                          
for axis in axes.flat:
  image = axis.imshow(nn.w[0].T[index].reshape(p_w, p_h),
                      cmap = plt.cm.gray, interpolation = 'nearest')
  axis.set_frame_on(False)
  axis.set_axis_off()
  index += 1
    
plt.show()
