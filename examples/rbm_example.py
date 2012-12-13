import sys
sys.path[:0] = '../../'

import pickle
import cProfile
from itertools import *

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
    y = ut.all_to_sparse( digits.target, max(digits.target) + 1 )
    X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y), "digits_rbm", True)
    X_val = np.vstack([X_val, X_test]) 
    y_val = np.vstack([y_val, y_test]) 

    hid_layer = 300

    bm = rbm.RBM(64, hid_layer)
    #exit()
    
    costs = bm.optimize(neur.mini_batch_generator(X), 2000, 0.08)
    print "validate squared_error",  bm.validate(X_val)
    #exit()

    filename = './random_set_cache/data_rbm_run.pkl'

    first_layer_weights = np.hstack([np.zeros((hid_layer,1)), bm.weights])
    #pickle.dump(first_layer_weights, open(filename, 'w'))

    # first_layer_weights = pickle.load(open(filename, 'r'))

    thetas  = neur.create_initial_thetas([64, hid_layer, 10], 0.12)
    thetas[0] =  first_layer_weights

    thetas, costs, val_costs = neur.gradient_decent_gen(izip(neur.mini_batch_generator(X, 10), 
                                                             neur.mini_batch_generator(y, 10)),
                                                        learning_rate = 0.05,
                                                        hidden_layer_sz = hid_layer,
                                                        iter = 8000,
                                                        thetas = thetas, 
                                                        X_val = X_val, 
                                                        y_val = y_val,
                                                        do_early_stopping = True)

    h_x, a = neur.forward_prop(X_test, thetas)
    print "percentage correct predictions: ", ut.percent_equal(ut.map_to_max_binary_result(h_x), y_test)
    print "training error:",   costs[-1:][0]
    print "validation error:", val_costs[-1:][0]
    print "lowest validation error:", min(val_costs)
    plt.plot(costs, label='cost')
    plt.plot(val_costs, label='val cost')
    plt.legend()
    plt.ylabel('error rate')
    plt.show()        

#    set = X[(np.random.rand(10) * len(X)).astype(int), :]
#    for x in set:
#        print '-- input --'
#        print (x > 0.1).astype(int).reshape(8,8)
#        print '-- input end --'
#        
#        print '-- output --'
#        print bm.get_impression(x).reshape(8,8)
#        print '-- output end --'
        
    


rbm_example()
