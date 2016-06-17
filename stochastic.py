import tensorflow as tf
import numpy as np

class StochasticGradientDescentState:
  def __init__(self, total_size, batch_size):
    self.rng = np.arange(total_size)
    self.batch_size = batch_size
    self.start, self.end = 0, batch_size

  def get_next_batch(self, data):
    batch = data[
      self.rng[self.start:self.end]
    ]
    self.start += self.batch_size
    self.end += self.batch_size

    if self.start >= len(self.rng):
      self.start, self.end = 0, self.batch_size
      np.random.shuffle(self.rng)

    return batch
