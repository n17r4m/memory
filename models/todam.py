import numpy as np
import math

class TODAM:
  def __init__(self, n, l, w0, L, N = 0, alpha = 0.8, delta = 0.2, lin = False):
    
    self.stop_on_error = False
    self.output_interference = False
    self.lin = lin
    
    self.n = n         # number of dimensions in feature vectors
    self.l = l         # lambda
    self.delta = delta
    self.alpha = alpha
    self.w0 = w0       # initial weight
    self.L = L         # number of items 
    self.LL = L + 2    # number of items (including start/end markers)
    self.N = N         # number of end competitors
    self.reset()
  
  def trial(self):
    self.k += 1
    for i in range(self.LL - 1):
      self.M = self.associate(self.items[i], self.items[i+1], i + 1)
  
  def reset(self):
    if self.lin:
      self.M = np.zeros(self.n*2 - 1)  # memory trace
    else:
      self.M = np.zeros(self.n)  # memory trace
    self.items = np.random.normal(0, 1.0/math.sqrt(n), (self.LL + N,n))  # items
    self.x = self.items[0]           # start token
    self.y = self.items[self.LL-1]   # end token
    self.k = 0
  
  def convolve(self, a, b):
    if self.lin:
      n = len(a)
      c = np.zeros((2*n - 1,))
      d = (len(c) - n) / 2
      for k in range(c.shape[0]):
        for i in range(n):
          for j in range(n):
            if i + j == k:
              c[k] += a[i] * b[j]
      return c;
    else:
      return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
  
  def correlate(self, a, b):
    if self.lin:
      
      if len(a) > len(b):
        a, b = b, a
      n = len(a)
      c = np.zeros((n,))
      for k in range(n):
        for i in range(len(a)):
          for j in range(len(b)):
            if j - i == k:
              c[k] += a[i] * b[j]
      return c # c[d:-d];
      # np.correlate(a, b, 'same')
    else:
      return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b).conj()).real
  
  def d(self, a):
    return self.M.dot(a)
  
  def g(self, j):
    return 1 - self.w(j)
  
  def w(self, j):
    if j is 1:
      return self.w0 * (math.e ** (self.lam() * (self.LL - 1)))
    else:
      return self.w0 * (math.e ** (self.lam() * (j - 2)))
  
  def lam(self):
    if self.lin:
      return self.l * math.e ** (-self.delta * (self.k - 1))
    else:
      return self.l
  
  def associate(self, a, b, j):
    if self.lin:
      pad = (2*len(a) - 1)/4
      item_mem = np.pad(self.g(j) * b, pad, 'constant')
      assoc_mem = self.w(j) * self.convolve(a, b)
      M = self.alpha * self.M + item_mem + assoc_mem
    else:
      d = (1 - self.d(a))
      print d
      
      item_mem =  self.g(j) * b
      assoc_mem =  self.w(j) * self.convolve(a, b)
      M = self.M + (1-self.d(item_mem)) * item_mem + (1-self.d(assoc_mem)) * assoc_mem
    return M
   
  def probe(self, a, j):
    b = self.correlate(self.M, a)
    if self.output_interference:
      self.M = self.associate(a, b, j)
    return b
  
  def deblur(self, a, j = 0):
    d = self.items.dot(a)
    d[:j] = -1e10
    i = np.argmax(d)
    return self.items[i], i
  
  def serial_anticipation(self, lin = False):
    r = np.zeros(self.LL)
    r[0] = 1
    for i in range(self.LL):
      j = i + 1
      f = self.items[i]
      g_ = self.probe(f, j)
      g, i_ = self.deblur(g_, j)
      #print f, g_, g
      if j == i_ and j < self.LL:
        r[j] = 1
    return r
  
  def recall(self, lin = False):
    r = np.zeros(self.LL)
    f = self.x
    r[0] = 1
    for j in range(1, self.LL):
      if lin:
        g_ = self.lin_probe(f, j)
      else: 
        g_ = self.probe(f, j)
      g, i_, d  = self.deblur(g_, j)
      #print j, i_
      if j == i_:
        r[j] = 1
        f = g
      elif self.stop_on_error:
        break;
      else:
        f = g_
    return r

"""
iterations = 100
num_items = 5
num_feats = 101 # must be odd
alpha = 0.859
lambd = 0.851
delta = 0.221
w0 = 0.828
competitors = 0

score_sum = np.zeros(num_items + 2)
"""

"""
t = TODAM(num_feats, lambd, w0, num_items, competitors, alpha, delta, True)
t.lin = True
t.output_interference = False
a = [1,2,3]
b = [2,3,4]

print "a", a
print "b", b

print "a*b.T"
print np.array([a]).T * (np.array([b]))

con = t.convolve(a, b)
print "convolve"
print con

print "con*b"
print np.array(con) * (np.array([b[::-1]]).T)

cor = t.correlate(con, b)
print "cor"
print cor
"""
"""
for i in range(iterations):
  t = TODAM(num_feats, lambd, w0, num_items, competitors, alpha, delta, False)
  t.lin = False
  t.output_interference = False
  t.trial()
  #t.trial()
  #t.trial()
  r = t.serial_anticipation()
  
  print r
  print t.M
  score_sum += r
  
  
print score_sum/iterations
"""

#print t.M

"""


for i in range(n):
  t = TODAM(400, 0.27, 1, 3)

  a = t.items[1]
  a_ = t.probe(t.x, 1)

  b = t.items[2]
  b_ = t.probe(a, 2)

  a_2, i, d = t.deblur(a_)

  if i == 1:
    c += 1.0

print t.items[0]

print c/n
"""


