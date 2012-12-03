import sys
sys.path[:0] = '../../'
import unittest
import rbm_bias as rbm
import neurpy as neur
import numpy as np
from sklearn import datasets

class TestRBM(unittest.TestCase):
    
    def setUp(self):
        self.rbm = rbm.RBM(3, 2)

        self.rbm.weights = np.array([ [0.778157,   0.118274,   0.264556],
                                      [0.870012,   0.639921,   0.774234]])

        self.visible_layer = np.array([[1,   1,   1],
                                       [1,   0,   1],
                                       [1,   0,   0],
                                       [0,   0,   1]])

        self.hidden_layer = np.array([[1,   1],   
                                      [0,   1],
                                      [0,   1],   
                                      [0,   0]])

        self.rbm.hidden_bias = np.array([0.1, 0.2])
        self.rbm.visible_bias = np.array([0.1, 0.2, 0.3])

    def test_expand_hidden_bias(self):
        res = self.rbm.expand_hidden_bias(3)
        expected = np.array([[0.1, 0.2],
                             [0.1, 0.2],
                             [0.1, 0.2]])
        self.assertTrue(self.all_equalish(res, expected))

    def test_expand_hidden_bias(self):
        res = self.rbm.expand_visible_bias(2)
        expected = np.array([[0.1, 0.2, 0.3],
                             [0.1, 0.2, 0.3]])
        self.assertTrue(self.all_equalish(res, expected))

    def test_prop_up(self):
        res = self.rbm.prop_up(self.visible_layer)
        expected = neur.sigmoid(np.array([[ 1.260987,  2.484167],
                                          [ 1.142713,  1.844246],
                                          [ 0.878157,  1.070012],
                                          [ 0.364556,  0.974234]]))
        self.assertTrue(self.all_equalish(res, expected))

    def test_prop_down(self):
        res = self.rbm.prop_down(np.array([[1, 0],
                                           [1, 1]]))
        expected = [[ 0.70644016,  0.57890356,  0.63750605],
                    [ 0.85172171,  0.72276027,  0.79229089]]
        self.assertTrue(self.all_equalish(res, expected))

    def test_goodness(self):
        res = self.rbm.goodness(self.visible_layer, self.hidden_layer)
        self.assertTrue(self.equalish(res, 2.014853))
   
    def test_goodness_gradient(self):
        res = self.rbm.goodness_gradient(self.visible_layer, self.hidden_layer)
        expected = [[0.25000,   0.25000,   0.25000],
                    [0.75000,   0.25000,   0.50000]]
        self.assertEqual(res.shape, self.rbm.weights.shape)
        self.assertTrue(self.all_equalish(res, expected))

    def test_goodness_gradient_layer(self):
        res = self.rbm.goodness_gradient_layer(self.visible_layer)
        expected = np.array([0.75,   0.25,   0.75])
        self.assertTrue(self.all_equalish(res, expected))
    
    def test_random_binary_sample(self):
        hid_prob = self.rbm.prop_up(self.visible_layer)
        res = self.rbm.random_binary_sample(hid_prob)

    def test_cd1(self):
        w_gradients, v_gradients, h_gradients, hid_prob = self.rbm.cd1(self.visible_layer)
        print w_gradients
        print v_gradients
        print h_gradients
        return True

# def test_prop_down(self):
#        res = self.rbm.prop_up(self.rbm.visible_layer)

#        self.assertTrue(self.equalish(res, 0.6946163218))
    def all_equalish(self, a, b, tol = 0.0001):
        dif = np.abs(a - b)
        return np.all(dif.flatten() < tol)

    def equalish(self, a, b, tol = 0.0001):
        return np.abs(a - b) < tol

if __name__ == '__main__':
    unittest.main()
