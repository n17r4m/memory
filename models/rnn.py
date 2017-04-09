#refs:
"""
http://philipperemy.github.io/keras-stateful-lstm/

"""

from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional

import numpy as np
import math


look_back = 1

#np.random.seed(4)

class Learner(object):
  def __init__(self, n, l, L, N = 0, state = True):
    print "New Learner"
    self.stop_on_error = False
    self.output_interference = True
  
    self.n = n         # number of dimensions in feature vectors
    self.l = l         # nodes per level
    self.L = L         # number of items 
    self.LL = L + 2    # number of items (including start/end markers)
    self.N = N         # number of end competitors
    self.M = np.zeros(self.n)  # memory trace
    self.shuffle_items()
    
    self.model = Sequential()
    if state:
      self.model.add(Bidirectional(SimpleRNN(l, return_sequences=True, stateful=state), batch_input_shape=(1,1,n)))
    else:
      self.model.add(Bidirectional(SimpleRNN(l, return_sequences=True), input_shape=(1,n)))
    #self.model.add(BatchNormalization())
    #self.model.add(Activation('tanh'))
    #self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Bidirectional(SimpleRNN(l, stateful=state)))
    #self.model.add(BatchNormalization())
    #self.model.add(LeakyReLU(alpha=0.2))
    #self.model.add(Activation('tanh'))
    self.model.add(Dense(n))
    self.model.compile(loss='cosine_proximity', optimizer='adam') #mse, #rmsprop
    self.initial_weights = self.model.get_weights()

  def shuffle_items(self):
    self.items = np.random.normal(0, 1.0/self.n, (self.LL, self.n))  # items
    self.x = self.items[0]    # start token
    self.y = self.items[-1]   # end token

  def reset(self, weights=None):
    # https://github.com/fchollet/keras/issues/341
    self.shuffle_items()
    if weights is None:
        weights = self.initial_weights
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    self.model.set_weights(weights)

  def seq_trial(self, epochs = 1, batch = 1):
    for e in range(epochs):
      self.model.reset_states()
      self.model.fit(
        self.items.reshape(self.LL, 1, self.n), 
        np.roll(self.items, -1, axis=0), 
        nb_epoch=1, batch_size=batch, shuffle=False, verbose=0)

  def deblur(self, a, j):
    opts = self.items[j:]
    d = opts.dot(a)
    i = np.argmax(d)
    return opts[i], i+j


  def serial_anticipation(self):
    r = np.zeros(self.LL)
    r[0] = 1.0
    for i in range(self.LL-1):
      j = i + 1
      f = self.items[i]
      g_ = self.probe(f, j)
      g, i_ = self.deblur(g_, j)
      if j == i_:
        r[j] = 1.0
    return r

  def probe(self, f, j):
    g_ = self.model.predict(f.reshape(1,1,self.n))[0]
    if self.output_interference:
      self.model.fit(
        self.items[j-1].reshape(1, 1, self.n), 
        g_.reshape(1,self.n), 
        nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
    g_ = self.model.predict(f.reshape(1,1,self.n))[0]
    return g_


compiles = 10
trials = 25
feats = 50
nodes = 40
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

#[ 1.     0.384  0.344  0.28   0.328  0.192  0.568  0.712  1.   ]
#[ 1.     0.232  0.16   0.328  0.36   0.472  0.336  0.192  1.   ]

#[ 1.     0.52   0.576  0.704  0.576  0.984  0.576  0.744  1.   ]
#[ 1.     0.568  0.744  0.472  0.568  0.712  0.88   0.936  1.   ]

#print serial_anticipation()
"""
print a
print a_
print a2
"""

