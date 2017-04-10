#refs:
"""
http://philipperemy.github.io/keras-stateful-lstm/

"""

from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

import numpy as np
import math


look_back = 1

#np.random.seed(4)

class Learner(object):
  def __init__(self, options):
    self.options = options
    n = options.n
    l = int(options.s)
    self.n = n                 # number of dimensions in feature vectors
    self.l = l                 # nodes per level
    self.L = options.N         # number of items 
    self.N = options.c         # number of end competitors
    self.LL = self.L + self.N + 2 # number of items (including start/end markers)
    self.length = self.LL      # for test api
    self.M = np.zeros(self.n)  # memory trace
    self.t = options.t         # recall threshold
    self.shuffle_items()
    
    self.model = Sequential()
    self.model.add(Bidirectional(GRU(l, return_sequences=True, stateful=True), batch_input_shape=(1,1,n)))
    #self.model.add(BatchNormalization())
    #self.model.add(Activation('tanh'))
    #self.model.add(Bidirectional(GRU(l, return_sequences=True, stateful=True)))
    self.model.add(Bidirectional(GRU(l, stateful=True)))    
    #self.model.add(Activation('tanh'))
    #self.model.add(Bidirectional(GRU(l, stateful=True)))
    #self.model.add(Activation('tanh'))
    #self.model.add(LeakyReLU(alpha=0.2))
    #self.model.add(BatchNormalization())
    #self.model.add(LeakyReLU(alpha=0.2))
    #self.model.add(Activation('tanh'))
    self.model.add(Dense(n))
    # defaults: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
    self.model.compile(loss='cosine_proximity', optimizer=adam) #mse, #rmsprop
    self.initial_weights = self.model.get_weights()

  def __len__(self):
    return self.L + 2

  def shuffle_items(self):
    self.items = np.random.normal(0, 1.0/self.n, (self.LL, self.n))  # items
    self.x = self.items[0]    # start token
    self.y = self.items[-(self.N+1)]   # end token

  def reset(self, weights=None):
    # https://github.com/fchollet/keras/issues/341
    self.shuffle_items()
    if weights is None:
        weights = self.initial_weights
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    self.model.set_weights(weights)

  def trial(self):
    for e in range(self.options.e):
      self.model.reset_states()
      self.model.fit(
        self.items.reshape(self.LL, 1, self.n), 
        np.roll(self.items, -1, axis=0), 
        nb_epoch=1, batch_size=1, shuffle=False, verbose=0)

  def deblur(self, a, j = 0):
    opts = self.items[j:]
    d = opts.dot(a)
    i = np.argmax(d)
    t = math.sqrt(a.dot(opts[i]) ** 2)
    if t < self.t:
      return None, -1
    return opts[i], i+j


  def probe(self, f, j):
    g_ = self.model.predict(f.reshape(1,1,self.n))[0]
    if self.options.oi:
      self.model.fit(
        self.items[j-1].reshape(1, 1, self.n), 
        g_.reshape(1,self.n), 
        nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
    g_ = self.model.predict(f.reshape(1,1,self.n))[0]
    return g_

"""
compiles = 10
trials = 25
feats = 50
nodes = 30
items = 9
r = np.zeros(items+2)

for c in range(compiles):
  learner = Learner(feats, nodes, items, 0, True)
  for i in range(trials):
    learner.shuffle_weights()
    learner.seq_trial(2, 1)
    sa = learner.serial_anticipation() 
    print sa
    r += sa
print r/(trials*compiles)
"""

#todo: reinit items each run

"""
compiles = 50
trials = 25
feats = 50
nodes = 30
items = 9
epochs = 2
batch = 1
state = True
self.model = Sequential()
Bidirectional(GRU(l, return_sequences=True, stateful=state), batch_input_shape=(1,1,n)))
Bidirectional(GRU(l, stateful=state)))
self.model.add(Dense(n))
loss='cosine_proximity', optimizer='adam') #mse, #rmsprop
"""
# [ 1.      0.7696  0.5344  0.4328  0.36    0.4352  0.3624  0.4344  0.4944  0.6232  1.    ]


"""
print a
print a_
print a2
"""

