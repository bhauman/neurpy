import sys
sys.path[:0] = '../../'
import cProfile

import unittest
import neurpy as neur
import numpy as np
import util as ut
from sklearn import datasets
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

def basic_gradient_descent():
    digits = datasets.load_digits()
        # iris = datasets.load_iris()
    X = digits.images.reshape((digits.images.shape[0], -1))

    scaler = pre.Scaler()
    X = scaler.fit_transform(X)
    
    y = ut.all_to_sparse( digits.target, max(digits.target) + 1 )
    X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y), "digits")
    X_val = np.vstack([X_val, X_test]) 
    y_val = np.vstack([y_val, y_test]) 
    thetas, costs, val_costs = neur.gradient_decent(np.array(X), 
                                                    np.array(y),
                                                    #hidden_layer_sz = 11,
                                                    hidden_layer_sz = 200,
                                                    iter = 500,
                                                    wd_coef = 0.0,
                                                    learning_rate = 0.25,
                                                    momentum_multiplier = 0.9,
                                                    rand_init_epsilon = 0.012,
                                                    do_early_stopping = True,
                                                    #do_dropout = True,
                                                    #dropout_percentage = 0.7,
                                                    #do_learning_adapt = True,
                                                    X_val = np.array(X_val),
                                                    y_val = np.array(y_val))
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

 
basic_gradient_descent()
#cProfile.run('basic_gradient_descent()', 'profile.stats')
#import pstats

#p = pstats.Stats('profile.stats')
#p.strip_dirs()
#p.sort_stats('time')
#p.print_stats()
