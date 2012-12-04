import sys
sys.path[:0] = '../../'
import unittest
import rbm
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

    def test_prop_up(self):
        res = self.rbm.prop_up(self.visible_layer)
        expected = [[ 0.76151,   0.90756],
                    [ 0.73937,   0.83811],
                    [ 0.68528,   0.70475],
                    [ 0.56576,   0.6844 ]]
        self.assertTrue(self.all_equalish(res, expected))

    def test_prop_down(self):
        res = self.rbm.prop_down([[1, 0],
                                  [1, 1]])
        expected = [[0.68528,   0.52953,   0.56576],
                    [0.83864,   0.68096,   0.73862]]
        self.assertTrue(self.all_equalish(res, expected))

    def test_goodness(self):
        res = self.rbm.goodness(self.visible_layer, self.hidden_layer)
        self.assertTrue(res, 1.489853)

    def test_free_energy(self):
        res = self.rbm.free_energy(self.visible_layer)
        print res
   
    def test_goodness_gradient(self):
        res = self.rbm.goodness_gradient(self.visible_layer, self.hidden_layer)
        expected = [[0.25000,   0.25000,   0.25000],
                    [0.75000,   0.25000,   0.50000]]
        self.assertEqual(res.shape, self.rbm.weights.shape)
        self.assertTrue(self.all_equalish(res, expected))
    
    def test_random_binary_sample(self):
        hid_prob = self.rbm.prop_up(self.visible_layer)
        res = self.rbm.random_binary_sample(hid_prob)

    def test_cd1(self):
        gradients, hid_prob = self.rbm.cd1(self.visible_layer)
        print gradients

# def test_prop_down(self):
#        res = self.rbm.prop_up(self.rbm.visible_layer)

#        self.assertTrue(self.equalish(res, 0.6946163218))
    def all_equalish(self, a, b):
        dif = np.abs(a - b)
        return np.all(dif.flatten() < 0.0001)

    def equalish(self, a,b):
        return np.abs(a - b) < 0.00001

if __name__ == '__main__':
    unittest.main()
