#!/usr/bin/env python

from argparse import ArgumentParser
from util.package_contents import package_contents
from util.mean_confidence_interval import mean_confidence_interval
import importlib
from util.Spinner import Spinner
import numpy as np

MODELS_AVAILABLE = package_contents("models")
TESTS_AVAILABLE = package_contents("tests")



def build_parser():
    parser = ArgumentParser()
    parser.add_argument('model', help='Memory model to use '+repr(MODELS_AVAILABLE))
    parser.add_argument('test', help='Test to execute '+repr(TESTS_AVAILABLE))
    parser.add_argument('-j', help='Number of iteration epochs', default=5, type=int)
    parser.add_argument('-i', help='Number of iterations in iteration epoch', default=25, type=int)
    parser.add_argument('-e', help='Number of training epochs', default=2, type=int)
    parser.add_argument('-n', help='Number of features', default=500, type=int)
    parser.add_argument('-N', help='Number of items', default=7, type=int)
    parser.add_argument('-s', help='Network size (width or # neurons)', default=30, type=int)
    parser.add_argument('-l', help='lambda', default=0.3, type=float)
    parser.add_argument('-d', help='delta', default=0.2, type=float)
    parser.add_argument('-c', help='competitors', default=1, type=int)
    parser.add_argument('-w', help='initial weight', default=1, type=int)
    parser.add_argument('-oi', help='output interference', default=True, action='store_true')
    parser.add_argument('-lin', help='linear convolution', default=False, action='store_true')
    
    return parser

if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    
    Learner = importlib.import_module("models." + options.model).Learner
    trial = importlib.import_module("tests."  + options.test).trial
    
    epochs = options.j
    iterations = options.i
    
    results = []
    print "Options", options
    for j in range(epochs):
      learner = Learner(options)
      
      spinner = Spinner()
      spinner.start()
      
      for i in range(iterations):
        learner.trial()
        results.append(trial(learner, options))
        learner.reset()
      
      spinner.stop()
      
      print np.mean(results, axis=0)
    
    results = np.array(results)
    ci = []
    for i in range(results.shape[1]):
      ci.append(mean_confidence_interval(results[:,i], confidence=0.95))

    for i in range(len(ci)):
      mean, lower, upper = str(ci[i][0]), str(ci[i][1]), str(ci[i][2])
      print mean + "\t" + lower + "\t" + upper
    
    #
    #  results
    #print mean_confidence_interval(results, confidence=0.95)
    
    
    
