import unittest
import neurpy as neur
import numpy as np
from sklearn import datasets
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt


class TestNuerpy(unittest.TestCase):
    def test_cross_validation_sets(self):
        r = np.arange(10)
        X = np.array([r,r * 2,r * 3,r * 4,r * 5,r * 6,r * 7,r * 8,r * 9,r * 10])
        self.assertEqual(X.shape, (10,10))
        y = np.array(map(lambda x: [x], r))
        self.assertTrue(np.equal(y,np.array(([0],[1],[2],[3],[4],[5],[6],[7],[8],[9]))).all)
        X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(X,y)
        self.assertEqual(X.shape, (8, 10))
        self.assertEqual(X_val.shape, (1, 10))
        self.assertEqual(X_test.shape, (1, 10))
        self.assertEqual(y.shape, (8, 1))
        self.assertEqual(y_val.shape, (1, 1))
        self.assertEqual(y_test.shape, (1, 1))   

    def test_rand_init_theta(self):
        res = neur.rand_init_theta(5, 4)
        self.assertEqual(res.shape, (4,6))
        self.assertTrue(np.mean(res) < 0.12/2)

    def test_sigmoid(self):
        res = neur.sigmoid(np.array(([1, 2], [3, 4])))
        target = np.array([[ 0.7310585,   0.88079708],
                           [ 0.95257413,  0.98201379]])
        self.equalish(res, target)
        
    def test_forward_prop(self):
        X = np.array(([1, 2,3], [2,3,4]))
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        
        out, a = neur.forward_prop(X, [theta, theta2])

        expected_result =  [[0.53407,   0.47487,   0.54053],
                            [0.53427,   0.47419,   0.54133]]
        self.equalish(out, expected_result)
        self.equalish(a[2], out)
        self.equalish(a[1], [[1.00000,   0.59179,   0.56096],
                             [1.00000,   0.61871,   0.58923]])
        self.equalish(a[0], [[1,   1,   2,   3],
                             [1,   2,   3,   4]])

        theta3 = np.array([[0.1007928,   0.1168322,  -0.0497762,  -0.0658923],
                           [-0.0841614,  -0.0378504,  -0.0918123,   0.0031022]])
        out, a = neur.forward_prop(X, [theta, theta2, theta3])
        expected_result = [[0.52596,   0.46349],
                           [0.52596,   0.46350]]
        self.equalish(out, expected_result)
        self.equalish(a[3], out)
        self.equalish(a[2], [[1.00000,   0.53407,   0.47487,   0.54053],
                             [1.00000,   0.53427,   0.47419,   0.54133]])
        self.equalish(a[1], [[1.00000,   0.59179,   0.56096],
                             [1.00000,   0.61871,   0.58923]])
        self.equalish(a[0], [[1,   1,   2,   3],
                             [1,   2,   3,   4]])

    def test_logistic_squared_distance(self):
        h_x = np.array([[0.52596,   0.46349],
                      [0.52596,   0.46350]])
        y = np.array([[1, 0],[1, 0]]) 
        res = neur.logistic_squared_distance(h_x, y)
        self.equalish_atom(res, 1.2652)
    
    def test_cost_function_weight_decay(self):
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        res = neur.cost_function_weight_decay(2, [theta, theta2], 1)
        self.equalish_atom(res, 0.016502)
    
    def test_logistic_squared_distance_with_wd(self):
        h_x = np.array([[0.52596,   0.46349],
                      [0.52596,   0.46350]])
        y = np.array([[1, 0],[1, 0]]) 
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        res = neur.logistic_squared_distance_with_wd(h_x, y, [theta, theta2], 1)
        self.equalish_atom(res, 1.2817118)

    def test_gradient_check(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        y = np.array([[0, 0, 1],[0, 0, 1]])
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        res = neur.gradient_check(X, y, [theta, theta2], neur.logistic_squared_cost_function)
        self.equalish(res[0], np.array([[ 0.00562904,  0.00841451,  0.01404355,  0.01967259],
                                [-0.02588239, -0.03870536, -0.06458776, -0.09047015]]))
        self.equalish(res[1], np.array([[ 0.5341706 ,  0.3233084 ,  0.30720236],
                                     [ 0.4745306 ,  0.28720531,  0.27289721],
                                     [-0.45907194, -0.2778482 , -0.26400618]]))

    def test_backprop(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        y = np.array([[0, 0, 1],[0, 0, 1]])
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        res, a = neur.forward_prop(X, [theta, theta2])
        grads = neur.backprop(a, y, [theta, theta2], 0)
        grads_check = neur.gradient_check(X, y, [theta, theta2], neur.logistic_squared_cost_function)
        self.equalish(grads[0], grads_check[0])
        self.equalish(grads[1], grads_check[1])

    def test_softmax(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                            [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        h_x, a = neur.forward_prop(X, [theta, theta2])
        res = neur.softmax(h_x)
        self.equalish(np.sum(res, axis=1), [1,1])

    def test_create_dropout_indices(self):
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                          [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        theta3 = np.array([[0.1007928,   0.1168322,  -0.0497762,  -0.0658923],
                           [-0.0841614,  -0.0378504,  -0.0918123,   0.0031022]])
        print neur.create_dropout_indices([theta, theta2, theta3])

    def test_dropout_thetas(self):
        theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                          [0.013256,   0.092686,  -0.070016,   0.093055]))
        theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        theta3 = np.array([[0.1007928,   0.1168322,  -0.0497762,  -0.0658923],
                           [-0.0841614,  -0.0378504,  -0.0918123,   0.0031022]])
        dropped_thetas, indices = neur.dropout_thetas([theta, theta2, theta3])
        print dropped_thetas
        dropped_thetas[0] = dropped_thetas[0] + 10
        dropped_thetas[1] = dropped_thetas[1] + 10
        dropped_thetas[2] = dropped_thetas[2] + 10
        theta_orig = theta.copy()
        theta2_orig = theta2.copy()
        theta3_orig = theta3.copy()
        thetas = neur.recover_dropped_out_thetas([theta, theta2, theta3], dropped_thetas, indices)

        equal_indices = list(set(range(theta_orig.shape[0])) - set(indices[0]))
        self.not_equalish(theta_orig[indices[0], :], thetas[0][indices[0], :])
        self.equalish(theta_orig[equal_indices, :], thetas[0][equal_indices, :])

        not_equal_indices = [0] + map(lambda x: x + 1,(indices[0]))
        equal_indices = list(set(range(theta_orig.shape[0] + 1)) - set(not_equal_indices))
        self.not_equalish(theta2_orig[:, not_equal_indices], thetas[1][:, not_equal_indices])
        self.equalish(theta2_orig[:, equal_indices], thetas[1][:, equal_indices])
        
        equal_indices = list(set(range(theta3_orig.shape[0])) - set(indices[1]))
        self.not_equalish(theta3_orig[indices[1], :], thetas[2][indices[1], :])
        self.equalish(theta3_orig[equal_indices, :], thetas[2][equal_indices, :])
        
        self.equalish(theta, thetas[0])
        self.equalish(theta2, thetas[1])
        self.equalish(theta3, thetas[2])

        

    def est_gradient_decsent(self):
        iris = datasets.load_iris()
        X = iris.data

        scaler = pre.Scaler()
        X = scaler.fit_transform(X)

        y = self.all_to_sparse( iris.target, max(iris.target) + 1 )
        X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y))
        thetas, costs, val_costs = neur.gradient_decent(np.array(X), 
                                                        np.array(y), 
                                                        np.array(X_val),
                                                        np.array(y_val))

    def est_mini_batch_gradient_descent(self):
        digits = datasets.load_digits()
        # iris = datasets.load_iris()
        X = digits.images.reshape((digits.images.shape[0], -1))

        scaler = pre.Scaler()
        X = scaler.fit_transform(X)

        y = self.all_to_sparse( digits.target, max(digits.target) + 1 )
        X, y, X_val, y_val, X_test, y_test = neur.cross_validation_sets(np.array(X), np.array(y))
        thetas, costs, val_costs = neur.mini_batch_gradient_decent(np.array(X), 
                                                                   np.array(y),
                                                                   hidden_layer_sz = 11,
                                                                   iter = 1000,
                                                                   wd_coef = 0.0001,
                                                                   learning_rate = 0.35,
                                                                   momentum_multiplier = 0.9,
                                                                   rand_init_epsilon = 0.000012,
                                                                   do_early_stopping = True,
                                                                   X_val = np.array(X_val),
                                                                   y_val = np.array(y_val))
        h_x, a = neur.forward_prop(X_test, thetas)
        print "percentage correct predictions: ", self.percent_equal(self.map_to_max_binary_result(h_x), y_test)
        print "training error:",   costs[-1:][0]
        print "validation error:", val_costs[-1:][0]
        print "lowest validation error:", min(val_costs)
        plt.plot(costs, label='cost')
        plt.plot(val_costs, label='val cost')
        plt.legend()
        plt.ylabel('error rate')
        plt.show()        
        
    def equalish(self,a,b):
        dif = np.abs(a - b)
        self.assertTrue(np.all(dif.flatten() < 0.00001))

    def not_equalish(self,a,b):
        dif = np.abs(a - b)
        self.assertFalse(np.all(dif.flatten() < 0.00001))

    def equalish_atom(self, a, b):
        self.assertTrue(abs(a - b) < 0.00001)

    def percent_equal(self, a, b):
        num_equal = sum(map(lambda x,y: 1 if np.equal(x,y).all() else 0, a, b))
        return float(num_equal) / len(a)

    def map_to_max_binary_result(self,h_x):
        maxes = map(max, h_x)
        res = []
        for i in range(len(maxes)):
           res.append(h_x[i].tolist().index(maxes[i]))
        return self.all_to_sparse(res, max(res) + 1)

    def convert_to_sparse(self, ex, num_class):
        res = [0] * num_class
        res[ex] = 1
        return res

    def all_to_sparse(self, exs, num_class):
        return map(self.convert_to_sparse, exs, [num_class] * len(exs))

if __name__ == '__main__':
    unittest.main()
