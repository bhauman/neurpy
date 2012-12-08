import sys
sys.path[:0] = '../../'
import unittest
import neurpy as neur
import numpy as np
import autoencoder as auto
from sklearn import datasets
from sklearn import preprocessing as pre

class TestAutoencode(unittest.TestCase):

    def setUp(self):
        self.enc = auto.Autoencoder(3, 2)
        self.theta = np.array(([0.040281,  -0.034031,   0.075200,   0.071569],
                               [0.013256,   0.092686,  -0.070016,   0.093055]))
        self.theta2 = np.array([[0.1150530,   0.1013294,  -0.0686610],
                                [-0.0459608,   0.0020356,  -0.0995257],
                                [0.0948434,   0.0686487,   0.0481420]])
        self.enc.encode_weights = self.theta
        self.enc.decode_weights = self.theta2

    
    def test_contractive_reg(self):
        out, a = self.enc.forward_prop(np.array([[1,0,1],[1,1,0]]))
        accum = 0
        h = a[1][0][1:]
        print 'h', h
        w = self.enc.encode_weights
        print 'w', w
        for i in range(len(h)):
            hh = (h[i] * (1 - h[i])) ** 2
            in_sum = 0
            for j in range(w.shape[1]):
                in_sum += w[i,j] ** 2
            accum += hh * in_sum
        
        res = self.enc.contractive_reg(np.array([a[1][0][1:]]))
        self.assertEqual(accum, res)

    def test_contractive_reg_gradient(self):
        X = np.array([[1, 0, 1], [1, 1, 0]])

        def cost_func(x, y, thetas, lamb):
            self.enc.encode_weights = thetas[0]
            out, a = self.enc.forward_prop(x)
            return self.enc.contractive_reg(a[1][:, 1:])

        self.enc.encode_weights = self.theta

        gradients = neur.gradient_check(X, X, [self.enc.encode_weights], cost_func)

        self.enc.encode_weights = self.theta
        out, a =    self.enc.forward_prop(X)
        grad_new =  self.enc.contractive_reg_gradient(a[1][:,1:]) # get rid of ones from argument
        self.assertTrue(((grad_new - gradients) < 0.00035).all())

if __name__ == '__main__':
    unittest.main()
