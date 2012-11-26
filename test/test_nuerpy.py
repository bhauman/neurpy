import sys
sys.path[:0] = '../../'
import unittest
import neurpy as neur
import numpy as np
from sklearn import datasets
from sklearn import preprocessing as pre

class TestNuerpy(unittest.TestCase):

    def setUp(self):
        self.theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                          [0.013256,   0.092686,  -0.070016,   0.093055]))
        self.theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        self.theta3 = np.array([[0.1007928,   0.1168322,  -0.0497762,  -0.0658923],
                           [-0.0841614,  -0.0378504,  -0.0918123,   0.0031022]])        

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
        out, a = neur.forward_prop(X, [self.theta, self.theta2])

        expected_result =  [[0.53407,   0.47487,   0.54053],
                            [0.53427,   0.47419,   0.54133]]
        self.equalish(out, expected_result)
        self.equalish(a[2], out)
        self.equalish(a[1], [[1.00000,   0.59179,   0.56096],
                             [1.00000,   0.61871,   0.58923]])
        self.equalish(a[0], [[1,   1,   2,   3],
                             [1,   2,   3,   4]])

        out, a = neur.forward_prop(X, [self.theta, self.theta2, self.theta3])
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
        res = neur.cost_function_weight_decay(2, [self.theta, self.theta2], 1)
        self.equalish_atom(res, 0.016502)
    
    def test_logistic_squared_distance_with_wd(self):
        h_x = np.array([[0.52596,   0.46349],
                      [0.52596,   0.46350]])
        y = np.array([[1, 0],[1, 0]]) 
        res = neur.logistic_squared_distance_with_wd(h_x, y, [self.theta, self.theta2], 1)
        self.equalish_atom(res, 1.2817118)

    def test_gradient_check(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        y = np.array([[0, 0, 1],[0, 0, 1]])
        res = neur.gradient_check(X, y, [self.theta, self.theta2], neur.logistic_squared_cost_function)
        self.equalish(res[0], np.array([[ 0.00562904,  0.00841451,  0.01404355,  0.01967259],
                                [-0.02588239, -0.03870536, -0.06458776, -0.09047015]]))
        self.equalish(res[1], np.array([[ 0.5341706 ,  0.3233084 ,  0.30720236],
                                     [ 0.4745306 ,  0.28720531,  0.27289721],
                                     [-0.45907194, -0.2778482 , -0.26400618]]))

    def test_backprop(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        y = np.array([[0, 0, 1],[0, 0, 1]])
        res, a = neur.forward_prop(X, [self.theta, self.theta2])
        grads = neur.backprop(a, y, [self.theta, self.theta2], 0)
        grads_check = neur.gradient_check(X, y, [self.theta, self.theta2], neur.logistic_squared_cost_function)
        self.equalish(grads[0], grads_check[0])
        self.equalish(grads[1], grads_check[1])

    def test_softmax(self):
        X = np.array([[1, 2, 3], [2, 3, 4]])
        h_x, a = neur.forward_prop(X, [self.theta, self.theta2])
        res = neur.softmax(h_x)
        self.equalish(np.sum(res, axis=1), [1,1])

    def test_create_dropout_indices(self):

        res = neur.create_dropout_indices([self.theta, self.theta2, self.theta3])
        self.assertEqual(3, len(res))
        self.assertEqual(Ellipsis, res[0][1])
        self.assertEqual(Ellipsis, res[1][0])
        self.assertEqual(Ellipsis, res[2][1])
        
    def test_dropout_indices_each(self):
        return False

    def test_dropout_thetas(self):

        dropped_thetas, indices = neur.dropout_thetas([self.theta, self.theta2, self.theta3])
        #print dropped_thetas
        dropped_thetas[0] = dropped_thetas[0] + 10
        dropped_thetas[1] = dropped_thetas[1] + 10
        dropped_thetas[2] = dropped_thetas[2] + 10
        theta_orig = self.theta.copy()
        theta2_orig = self.theta2.copy()
        theta3_orig = self.theta3.copy()
        thetas = neur.recover_dropped_out_thetas([self.theta, self.theta2, self.theta3], dropped_thetas, indices)

        equal_indices = list(set(range(theta_orig.shape[0])) - set(indices[0][0]))
        self.not_equalish(theta_orig[indices[0][0], :], thetas[0][indices[0][0], :])
        self.equalish(theta_orig[equal_indices, :], thetas[0][equal_indices, :])

        not_equal_indices = indices[1][1]
        equal_indices = list(set(range(theta_orig.shape[0] + 1)) - set(not_equal_indices))
        self.not_equalish(theta2_orig[:, not_equal_indices], thetas[1][:, not_equal_indices])
        self.equalish(theta2_orig[:, equal_indices], thetas[1][:, equal_indices])

        not_equal_indices = indices[2][0]        
        equal_indices = list(set(range(theta3_orig.shape[0])) - set(not_equal_indices))
        self.not_equalish(theta3_orig[not_equal_indices, :], thetas[2][not_equal_indices, :])
        self.equalish(theta3_orig[equal_indices, :], thetas[2][equal_indices, :])
        
        self.equalish(self.theta, thetas[0])
        self.equalish(self.theta2, thetas[1])
        self.equalish(self.theta3, thetas[2])

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

    def equalish(self,a,b):
        dif = np.abs(a - b)
        self.assertTrue(np.all(dif.flatten() < 0.00001))

    def not_equalish(self,a,b):
        dif = np.abs(a - b)
        self.assertFalse(np.all(dif.flatten() < 0.00001))

    def equalish_atom(self, a, b):
        self.assertTrue(abs(a - b) < 0.00001)



if __name__ == '__main__':
    unittest.main()
