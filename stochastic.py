import tensorflow as tf
import numpy as np

def stochastic_batch(data, batch_size):
  is_multi = isinstance(data, (list, tuple, set))
  total_size = len(data[0]) if is_multi else len(data)

  if batch_size > total_size:
    while True:
      yield data

  start, end = 0, batch_size
  rng = np.arange(total_size)

  while True:
    batch = tuple(x[rng[start:end]] for x in data) if is_multi else data[rng[start:end]]
    yield batch

    start += batch_size
    end += batch_size

    if start >= len(data):
      start, end = 0, batch_size
      np.random.shuffle(rng)
