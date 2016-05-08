import numpy as np
import matplotlib.pyplot as plt
__all__ = ['show_tiles']

TILE_BORDER_SIZE = 2

def show_tiles(tiles, nrows, ncols, **plot_kwargs):
  #This is waaaaay faster than subplot!
  tiled_image = np.ndarray(
    [nrows * (tiles.shape[1] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE, ncols * (tiles.shape[2] + TILE_BORDER_SIZE) - TILE_BORDER_SIZE]
  )
  tiled_image.fill(0.25)

  for y in range(nrows):
    for x in range(ncols):
      i = ncols * y + x
      tile = tiles[i] if i < len(tiles) else np.zeros(tiles.shape[1:])

      y_offset = y * (tiles.shape[1] + TILE_BORDER_SIZE)
      x_offset = x * (tiles.shape[2] + TILE_BORDER_SIZE)

      tiled_image[y_offset:y_offset + 28, x_offset:x_offset + 28] = tile

  plt.imshow(tiled_image, **plot_kwargs)
  plt.show()

def sample_random_patches(imgs, n, w, h):
  pass
