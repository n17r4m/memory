# Serial Anticipation

import numpy as np


def trial(learner, options):
  list_length = len(learner) 
  result = np.zeros(list_length)
  result[0] = 1.0
  for i in range(list_length - 1):
    j = i + 1
    f = learner.items[i]
    g_ = learner.probe(f, j)
    g, j_ = learner.deblur(g_, j)
    if j == j_:
      result[j] = 1.0
  return result
