from scipy.io import loadmat
from PIL import Image
from gbl import pil2ndarray

import random as rnd
import numpy as np

#Sampling mainly for the sparse autoencoder exercise.
__all__=('sample_patches', 'sample_mat_patches')

def normalize(data, mean, std):
  return 0.4 * (np.maximum(np.minimum(data - mean, std), -std) / std + 1) + 0.1

def sample_patches(w, h, flat=True):
  imgs = np.array([pil2ndarray(Image.open('res/{0}.jpg'.format(x + 1)).convert('L')) for x in range(10)])

  patches = []
  for i in range(10000):
    a, b = rnd.randint(0, imgs[0].shape[0] - w), rnd.randint(0, imgs[0].shape[1] - h)
    patches.append(imgs[rnd.randint(0, len(imgs) - 1), b:b + h,a:a+w])

  patches = np.array(patches)
  mean = patches.mean()
  std = 3 * patches.std()

  #Must normalize images using training set values
  imgs = normalize(imgs, mean, std)
  patches = normalize(patches, mean, std)

  return imgs, patches.reshape([10000, w * h]) if flat else patches

def sample_mat_patches(w, h, flat=True):
  imgs = loadmat('res/IMAGES.mat', mat_dtype=True)['IMAGES'].swapaxes(2,1).swapaxes(1,0)

  patches = []
  for i in range(10000):
    a, b = rnd.randint(0, imgs.shape[2] - w), rnd.randint(0, imgs.shape[1] - h)
    patch = imgs[rnd.randint(0, imgs.shape[0] - 1), b:b + h, a:a + w]
    patches.append(patch)

  patches = np.array(patches)
  mean = patches.mean()
  std = 3 * patches.std()

  #Must normalize images using training set values
  imgs = normalize(imgs, mean, std)
  patches = normalize(patches, mean, std)

  return imgs, patches.reshape([10000, w * h]) if flat else patches
