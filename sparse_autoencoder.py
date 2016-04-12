import numpy as np
import scipy.optimize
from sys import stdout as cout

# Logistic neural net with support for sparse autoencoding
class SimpleSAE:
  def __init__(self, layers, decay = 0, sparsity = 0, sparsity_penalty = 0):
    self.s = layers.copy() #Input layer, hidden layers, and output layer sizes
    self.l = decay # l for lambda
    self.p = sparsity
    self.pb = sparsity_penalty

  def flatten_weights(self, w, b):
    return np.concatenate([w_.flatten() for w_ in w] + [b_.flatten() for b_ in b])

  def unflatten_weights(self, w):
    s = self.s
    w_sizes = [s[i] * s[i + 1] for i in range(len(s) - 1)]
    b_start_index = sum(w_sizes)

    def prev_w(i):
      return 0 if i == 0 else w_sizes[i-1]

    def prev_b(i):
      return 0 if i == 0 else s[i]

    return ([
      w[prev_w(i):prev_w(i) + w_sizes[i]].reshape(s[i], s[i + 1])\
      for i in range(len(s) - 1)
    ], [
      w[b_start_index + prev_b(i):b_start_index + prev_b(i) + s[i + 1]].reshape(s[i + 1])\
      for i in range(len(s) - 1)
    ])

  #I find having column vectors for the weights to be somewhat more elegant
  def get_initial_weights(self, flat=False):
    w = [np.random.uniform(-self.get_rand_range(i), self.get_rand_range(i), [self.s[i + 1], self.s[i]]) for i in range(len(self.s) - 1)]
    b = [np.zeros(self.s[i + 1]) for i in range(len(self.s) - 1)]
    return (w, b) if not flat else self.flatten_weights(w, b)
    
  def get_rand_range(self, i):
    return np.sqrt(6 / (np.sum(self.s[1:]) + 1))

  def optimize(self, x, y, max_iter, **kwargs):
    info = {'i': 0}
    w = scipy.optimize.minimize(
      self.cost, self.get_initial_weights(flat=True),
      method='L-BFGS-B',
      args=(x, y, info),
      jac=True,
      options = {
        'maxiter': max_iter
      }
    )
    print(w.message, 'Success:', w.success)
    self.w, self.b = self.unflatten_weights(w.x)

  def feed_forward(self, x, a = None, w = None, b = None):
    w, b = w if w is not None else self.w, b if b is not None else self.b
    a_ = x
    for i in range(len(w)):
      a_ = np.dot(a_, w[i]) + b[i]
      a_ = 1 / (1 + np.exp(-a_))

      if a is not None:
        a.append(a_)

    return a_

  def cost(self, theta, x, y, info):
    w, b = self.unflatten_weights(theta)
    p, pb = self.p, self.pb
    a = []
    jp = 0

    self.feed_forward(x, a, w, b)

    inv_m = 1 / x.shape[0]
    diff = (a[-1] - y)
    d_ = diff * (a[-1] * (1 - a[-1]))
    djdw = [inv_m * np.dot(a[-2].T if len(a) > 1 else x.T, d_) + self.l * w[-1]]
    djdb = [inv_m * d_.sum(0)]

    for i in range(-2, -len(w) - 1, -1):
      kl_grad = 0

      if p and pb:
        p_ = inv_m * a[i].sum(0)
        p_a = p / p_
        p_b = (1 - p) / (1 - p_)
        kl_grad = pb * (p_b - p_a)
        jp += pb * np.sum(p * np.log(p_a) + (1 - p) * np.log(p_b))

      d_ = (np.dot(d_, w[i + 1].T) + kl_grad) * (a[i] * (1 - a[i]))

      djdw.append(inv_m * np.dot(a[i - 1].T if i > -len(w) else x.T, d_) + self.l * w[i])
      djdb.append(inv_m * d_.sum(0))

    cost = 0.5 * inv_m * (diff * diff).sum() + 0.5 * self.l * np.sum((w_ * w_).sum() for w_ in w) + jp

    if info.get('i') % 5 == 0:
      print(cost, info.get('i', 0))

    info['i'] += 1

    djdw.reverse()
    djdb.reverse()
    return cost, self.flatten_weights(djdw, djdb)
