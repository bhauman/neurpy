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

def basic_gradient_descent():
    data = np.genfromtxt('./stack_data_1000.csv', delimiter=',')
    X = data[:,:-1]
    y = data[:,-1:]

    scaler = pre.Scaler()
    X_front = scaler.fit_transform(X[:,0:6])
    X[:,0:6] = neur.sigmoid(X_front)

    y = np.array(map(lambda x: [0, 1] if x == 0 else [1, 0], y.flatten()))

    X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y), "basic_kaggle_data", True)
    X_val = np.vstack([X_val, X_test]) 
    y_val = np.vstack([y_val, y_test])


    hid_layer = 50

    
    #bm = rbm.RBM(X.shape[1], 50)
    #costs = bm.optimize(X, 1000, 0.05, val_set = X_val)


    thetas, costs, val_costs = neur.gradient_decent(np.array(X), 
                                                    np.array(y),
                                                    #hidden_layer_sz = 11,
                                                    hidden_layer_sz = 50,
                                                    iter = 2000,
                                                    wd_coef = 0.0,
                                                    learning_rate = 0.015,
                                                    momentum_multiplier = 0.9,
                                                    rand_init_epsilon = 0.012,
                                                    do_early_stopping = True,
                                                    do_dropout = True,
                                                    dropout_percentage = 0.5,
                                                    #do_learning_adapt = True,
                                                    X_val = np.array(X_val),
                                                    y_val = np.array(y_val))
    h_x, a = neur.forward_prop(X_val, thetas)
    binary_result = ut.map_to_max_binary_result(h_x)
    print "percentage correct predictions: ", ut.percent_equal(binary_result, y_val)
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
