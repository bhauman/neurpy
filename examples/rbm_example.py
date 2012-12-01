import sys
sys.path[:0] = '../../'
import cProfile

import unittest
import neurpy as neur
import numpy as np
import rbm
import util as ut
from sklearn import datasets
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

def rbm_example():
    digits = datasets.load_digits()
    X = digits.images.reshape((digits.images.shape[0], -1))
    X = (X / 16.0)

    bm = rbm.RBM(64, 70)

    #exit()
    
    costs = bm.optimize(X, 1000, 0.02)

    set = X[(np.random.rand(10) * len(X)).astype(int), :]
    for x in set:
        print '-- input --'
        print (x > 0.1).astype(int).reshape(8,8)
        print '-- input end --'
        
        print '-- output --'
        print bm.get_impression(x).reshape(8,8)
        print '-- output end --'
        
    
    print "validate squared_error",  bm.validate(X)

rbm_example()
