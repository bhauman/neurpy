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
    data = np.genfromtxt('./stack_data_wide_val.csv', delimiter=',')
    X = data[:,:-1]
    y = data[:,-1:]

    scaler = pre.Scaler()
    X_val = scaler.fit_transform(X)

    y_val = np.array(map(lambda x: [0, 1] if x == 0 else [1, 0], y.flatten()))
    
    #X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y), "basic_kaggle_data", True)
    #X_val = np.vstack([X_val, X_test]) 
    #y_val = np.vstack([y_val, y_test])

    hid_layer = 300

    mg = neur.split_xy(neur.mini_batch_gen_from_file('stack_data_wide_train.csv', 40), 
                       -1,
                       apply_x = lambda x: scaler.transform(x.astype(float)),
                       apply_y = lambda y: np.array(map(lambda x: [0, 1] if x == 0 else [1, 0], y.flatten()))
                       )

    #bm = rbm.RBM(13408, hid_layer)
    #costs = bm.optimize(neur.just_x(mg), 1000, 0.0007, val_set = X_val)

    #first_layer_weights = np.hstack([np.zeros((hid_layer,1)), bm.weights])
    #thetas  = neur.create_initial_thetas([64, hid_layer, 2], 0.12)
    #thetas[0] =  first_layer_weights

    # best so far minibatchsize 40 hidden layer 100 learning rate 0.01

    thetas, costs, val_costs = neur.gradient_decent_gen(mg, 
                                                    #hidden_layer_sz = 11,
                                                    hidden_layer_sz = hid_layer,
                                                    iter = 20000,
                                                    wd_coef = 0.0,
                                                    learning_rate = 0.01,
                                                    #thetas = thetas,
                                                    momentum_multiplier = 0.9,
                                                    rand_init_epsilon = 0.0012,
                                                    do_early_stopping = True,
                                                    #do_dropout = True,
                                                    #dropout_percentage = 0.5,
                                                    #do_learning_adapt = True,
                                                    X_val = np.array(X_val),
                                                    y_val = np.array(y_val)
                                                    )
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
