import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
__all__ = ['show_tiles']

'''Image loading/display helper functions'''

#Return ndarray of (width, height) from PIL image
def pil2ndarray(img):
  '''Given an image loaded by pillow, returns an ndarray of intensity values.

  Args:
    img (PIL.Image): Image loaded by PIL.Image.open()

  Returns:
    (numpy array): A numpy array with intensity values taken from the PIL image
  '''

  #Must reshape flat array to correct size
  return np.array(img.getdata(), np.float32).reshape(img.size[0], img.size[1])

def plot_image(img, filename=None, show=True, **plot_kwargs):
  '''Quickly plot/save an image from a numpy array.
    
    Args:
      img (np.array): Image intensity values, which can include channels.
      filename (string, None): Optionally save the image to specified file.
      show (bool, True): Show the image using matplotlib.
      **plot_kwargs: Identical to matplotlib's imshow().
  '''
  if show:
    plt.imshow(img, **plot_kwargs)
    plt.axis('off')
    plt.show()

  if filename:
    Image.fromarray((img * 255).astype(np.uint8))\
      .save(filename)
  
def get_tile_image(tiles, nrows, ncols, normalize=True):
  '''Constructs tile image.

  Creates a 2D array that arranges tiles over nrows and ncols, as a single 
  image. The result can be fed into matplotlib's .imshow() function.

  Args:
    normalize (bool, True): Normalize to be between [0, 1].
    tiles (numpy.ndarray): 3-dimensional array of image tiles (image #, tile height, tile width).
    nrows (int): Number of rows of the overall tile grid.
    ncols (int): Number of columns of the overall tile grid.

  Returns:
    (np.array): A numpy array containing intensity values of the tiled image.
  '''

  TILE_BORDER_SIZE = 1

  #This is waaaaay faster than subplot! Basically create the dimensions of one 
  #large image that has enough resolution to show tiles given # of rows and # of 
  #columns

  # Generalize to include a 'channel' dimension
  if len(tiles.shape) == 3:
    tiles = tiles.reshape(tiles.shape + (1,))


  if normalize:
    tiles -= tiles.mean()

    for j in range(tiles.shape[0]):
      for i in range(tiles.shape[-1]):
        t_max = np.abs(tiles[j,:,:,i]).max()
        tiles[j,:,:,i] = ((tiles[j,:,:,i] / t_max) + 1) / 2

  tiled_image = np.zeros([
      nrows * (tiles.shape[1] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE, #Last row/column doesn't need a border
      ncols * (tiles.shape[2] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE,
      tiles.shape[3]
  ])

  for y in range(nrows):
    for x in range(ncols):
      i = ncols * y + x
      if i >= len(tiles):
        break
      tile = tiles[i]

      y_offset = y * (tiles.shape[1] + TILE_BORDER_SIZE)
      x_offset = x * (tiles.shape[2] + TILE_BORDER_SIZE)

      tiled_image[y_offset:y_offset + tiles.shape[1], x_offset:x_offset + tiles.shape[2],:] = tile

  return tiled_image.squeeze()

def sample_random_patches(imgs, n, w, h):
  '''Given a series of images, samples n patches of width w and height h.

  Args:
    imgs (numpy.ndarray): 3-dimensional array of image tiles (image #, tile width, tile height).
    n (int): Number of patches to sample.
    w (int): Width of the sample patch.
    h (int): Height of the sample patch.

  Returns:
    (numpy.array): 3-dimensional array of sampled patches (image #, patch height, patch width).
  '''

  patches = np.zeros([n, w, h])

  for i in range(n):
    a, b = (
      np.random.randint(0, imgs.shape[1] - h), 
      np.random.randint(0, imgs.shape[2] - w)
    )

    patches[i] = imgs[np.random.randint(0, imgs.shape[0]), b:b + h, a:a + w]

  return patches
