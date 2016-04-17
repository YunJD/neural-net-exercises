from scipy.io import loadmat
from PIL import Image
import random as rnd
import numpy as np

__all__=('pil2ndarray', 'sample_patches', 'sample_mat_patches')

# Return ndarray of (width, height) from PIL image
def pil2ndarray(img):
  # Must reshape flat array to correct size
  return np.array(img.getdata(), np.float32).reshape(img.size[0], img.size[1])

def sample_patches(w, h, flat=True):
  imgs = np.array([pil2ndarray(Image.open('res/{0}.jpg'.format(x + 1)).convert('L')) for x in range(10)])

  # Instead of applying normalization to the patches, apply it to the images
  istd = 3 * imgs.std()
  imean = imgs.mean()
  imgs = np.maximum(np.minimum(imgs - imean, istd), -istd) / istd
  imgs = (imgs + 1) * 0.4 + 0.1

  patches = []
  for i in range(10000):
    a, b = rnd.randint(0, imgs[0].shape[0] - w), rnd.randint(0, imgs[0].shape[1] - h)
    patches.append(imgs[rnd.randint(0, len(imgs) - 1), b:b + h,a:a+w])

  patches = np.array(patches)

  return imgs, patches.reshape([10000, w * h]) if flat else patches

def sample_mat_patches(w, h, flat=True):
  imgs = loadmat('res/IMAGES.mat', mat_dtype=True)['IMAGES'].swapaxes(2,1).swapaxes(1,0)

  # Instead of applying normalization to the patches, apply it to the images
  istd = 3 * imgs.std()
  imean = imgs.mean()
  imgs = np.maximum(np.minimum(imgs - imean, istd), -istd) / istd
  imgs = (imgs + 1) * 0.4 + 0.1

  patches = []
  for i in range(10000):
    a, b = rnd.randint(0, imgs.shape[2] - w), rnd.randint(0, imgs.shape[1] - h)
    patch = imgs[rnd.randint(0, imgs.shape[0] - 1), b:b + h, a:a + w]
    patches.append(patch)

  patches = np.array(patches)

  return imgs, patches.reshape([10000, w * h]) if flat else patches
