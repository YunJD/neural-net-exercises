from PIL import Image
import random as rnd
import numpy as np

__all__=('pil2ndarray', 'sample_patches')

# Return ndarray of (width, height) from PIL image
def pil2ndarray(img):
  # Must reshape flat array to correct size
  return np.array(img.getdata(), np.float32).reshape(img.size[0], img.size[1])

def sample_patches(w, h, flat=True):
  imgs = [pil2ndarray(Image.open('res/{0}.jpg'.format(x + 1)).convert('L')) for x in range(10)]
  imgs = np.array(imgs)

  # It is important that the shape be maintained in order to sample a 'patch'
  patches = []
  for i in range(10000):
    img = imgs[rnd.randint(0, len(imgs) - 1)]
    a, b = rnd.randint(0, img.shape[0] - w), rnd.randint(0, img.shape[1] - h)
    patches.append(img[a:a+w, b:b + h] / 255.0)

  patches = np.array(patches)
  return imgs, patches.reshape([10000, w * h]) if flat else patches
