import numpy as np

# Softmax only
class SimpleNN:
  EPS=0.0001 # Epsilon value fed into whatever random distribution to sample from

  def __init__(self, layers, decay = 0):
    self.n = layers[0] # Number of input nodes
    self.s = layers[1:-1] # Hidden layer node counts
    self.k = layers[-1] # output layer, i.e. number of classes
    self.l = decay # l for lambda

    # Layer 1 weights
    if len(self.s):
      self.w = [np.random.normal(0, self.EPS, [self.n, self.s[0]])]
      self.b = [np.random.normal(0, self.EPS, [self.s[0]])]

      for i in range(1, len(self.s)):
        self.w.append(
          np.random.normal(0, self.EPS, [self.s[i-1], self.s[i]])
        )
        self.b.append(
          np.random.normal(0, self.EPS, [self.s[i]])
        )
    else:
      self.w, self.b = [], []

    self.w.append(np.random.normal(0, self.EPS, [self.s[-1] if len(self.s) else self.n, self.k]))
    self.b.append(np.random.normal(0, self.EPS, [self.k]))

  def feed_forward(self, x, a = None, z = None):
    a_ = x

    for i in range(len(self.w) - 1):
      a_ = np.dot(a_, self.w[i]) + self.b[i]

      if z is not None:
        z.append(a_)

      a_ = 1. / (1. + np.exp(a_))

      if a is not None:
        a.append(a_)

    a_ = np.dot(a_, self.w[-1]) + self.b[-1]

    if z is not None:
      z.append(a_)

    a_ = np.exp(a_)
    a_ = a_ / (a_.sum(-1, keepdims=True))

    if a is not None:
      a.append(a_)

    return a_

  def back_propagation(self, x, y, alpha = 0.01):
    # x - input, single row vector or multiple row vectors
    # y - output, single row vector or multiple row vectors
    # alpha - learning rate

    a = []
    z = []
    self.feed_forward(x, a, z)

    inv_m = 1. / len(x)
    d_ = (a[-1] - y)

    # The softmax layer needs to be multiplied out in this way! Don't believe ufldl!
    djdw = np.dot(a[-2].T if len(a) > 1 else x.T, d_)
    djdb = d_.sum(0)

    # Going backwards, skipping output layer
    for i in range(-2, -len(self.w) - 2, -1):
      if i > -len(self.w) - 1:
        df = a[i] * (1. - a[i])
        d_ = np.dot(d_, self.w[i + 1].T) * -df

      self.w[i + 1] -= alpha * (inv_m * djdw + self.l * self.w[i + 1])
      self.b[i + 1] -= alpha * inv_m * djdb

      djdw = np.dot(a[i - 1].T if i > -len(self.w) else x.T, d_)
      djdb = d_.sum(0)

    return -np.sum(np.log(a[-1]) * y) * inv_m + self.l * 0.5 * sum(np.sum(np.power(x,2)) for x in self.w)
