import sys
sys.path[:0] = '../../'

import pickle
import cProfile

import unittest
import neurpy as neur
import numpy as np
import rbm #_bias as rbm
import util as ut
from sklearn import datasets
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

def rbm_mnist_example():
    mnist_train   = np.fromfile('mnist_training.csv', sep=" ")
    mnist_train   = np.array(mnist_train.reshape(256, 1000)).transpose()
    
    mnist_targets   = np.fromfile('mnist_training_targets.csv', sep=" ")
    mnist_targets   = np.array(mnist_targets.reshape(10, 1000)).transpose()
    
    X = mnist_train
    y = mnist_targets
    X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y), "digits_rbm_mnist")
    X_val = np.vstack([X_val, X_test]) 
    y_val = np.vstack([y_val, y_test]) 

    hid_layer = 300

    bm = rbm.RBM(256, hid_layer)
    #exit()
    
    #costs = bm.optimize(X, 370, 0.2, val_set = X_val)
    print "validate squared_error",  bm.validate(X_val)

    filename = './random_set_cache/data_rbm_run_without_bias.pkl'

    #first_layer_weights = np.hstack([np.zeros((hid_layer,1)), bm.weights]) # without bias
    #first_layer_weights = np.hstack([bm.hidden_bias.reshape(hid_layer, 1), bm.weights]) # with bias

    #pickle.dump(first_layer_weights, open(filename, 'w'))

    #exit()

    first_layer_weights = pickle.load(open(filename, 'r'))
    
    thetas  = neur.create_initial_thetas([64, hid_layer, 10], 0.12)
    thetas[0] =  first_layer_weights

    thetas, costs, val_costs = neur.gradient_decent(X, y,
                                                    learning_rate = 0.05,
                                                    hidden_layer_sz = hid_layer,
                                                    iter = 3000,
                                                    thetas = thetas,
                                                    X_val = X_val, 
                                                    y_val = y_val,
                                                    do_dropout = True,
                                                    dropout_percentage = 0.7,
                                                    do_early_stopping = True)

    h_x, a = neur.forward_prop(X_val, thetas)
    print "percentage correct predictions: ", ut.percent_equal(ut.map_to_max_binary_result(h_x), y_val)
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

rbm_mnist_example()
