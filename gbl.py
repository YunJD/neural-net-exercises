import numpy as np
import matplotlib.pyplot as plt
__all__ = ['show_tiles']

'''Image loading/display helper functions'''

#Return ndarray of (width, height) from PIL image
def pil2ndarray(img):
  '''Given an image loaded by pillow, returns an ndarray of intensity values.

  Args:
    img (PIL.Image): Image loaded by PIL.Image.open()
  '''

  #Must reshape flat array to correct size
  return np.array(img.getdata(), np.float32).reshape(img.size[0], img.size[1])

def show_tiles(tiles, nrows, ncols, **plot_kwargs):
  '''Displays tiles over nrows rows and ncols columns.  plot_kwargs are identical to 
  matplotlib.pyplot.imshow kwargs.

  Args:
    tiles (numpy.ndarray): 3-dimensional array of image tiles (image #, tile width, tile height).
    nrows (int): Number of rows of the overall tile grid
    ncols (int): Number of columns of the overall tile grid
    **plot_kwargs: **kwargs for matplotlib.pyplot.imshow
  '''

  TILE_BORDER_SIZE = 2

  #This is waaaaay faster than subplot! Basically create the dimensions of one 
  #large image that has enough resolution to show tiles given # of rows and # of 
  #columns
  tiled_image = np.ndarray([
      nrows * (tiles.shape[1] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE, #Last row/column doesn't need a border
      ncols * (tiles.shape[2] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE
  ])
  tiled_image.fill(plot_kwargs.get('vmin', np.min(tiles)))

  for y in range(nrows):
    for x in range(ncols):
      i = ncols * y + x
      if i >= len(tiles):
        break
      tile = tiles[i]

      y_offset = y * (tiles.shape[1] + TILE_BORDER_SIZE)
      x_offset = x * (tiles.shape[2] + TILE_BORDER_SIZE)

      tiled_image[y_offset:y_offset + tiles.shape[1], x_offset:x_offset + tiles.shape[2]] = tile

  plt.imshow(tiled_image, **plot_kwargs)
  plt.show()

def sample_random_patches(imgs, n, w, h):
  '''Given a series of images, samples n patches of width w and height h.

  Args:
    imgs (numpy.ndarray): 3-dimensional array of image tiles (image #, tile width, tile height)
    n (int): Number of patches to sample
    w (int): Width of the sample patch
    h (int): Height of the sample patch

  Returns:
    numpy.ndarray: 3-dimensional array of sampled patches (image #, patch width, patch height)
  '''

  patches = np.zeros([n, w, h])

  for i in range(n):
    a, b = (
      np.random.randint(0, imgs.shape[1] - h), 
      np.random.randint(0, imgs.shape[2] - w)
    )

    patches[i] = imgs[np.random.randint(0, imgs.shape[0]), b:b + h, a:a + w]

  return patches
