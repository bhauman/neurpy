import sys
sys.path[:0] = '../'
import unittest

import gnumpy as gpu
import numpy as np
import neurpy_gpu as neur

from sklearn import datasets
from sklearn import preprocessing as pre

class TestNuerpyGpu(unittest.TestCase):

    def setUp(self):
        self.theta = gpu.as_garray(([0.040281,  -0.034031,   0.075200,   0.071569],
                                    [0.013256,   0.092686,  -0.070016,   0.093055]))
        self.theta2 = gpu.as_garray([[0.1150530,   0.1013294,  -0.0686610],
                           [-0.0459608,   0.0020356,  -0.0995257],
                           [0.0948434,   0.0686487,   0.0481420]])
        self.theta3 = gpu.as_garray([[0.1007928,   0.1168322,  -0.0497762,  -0.0658923],
                           [-0.0841614,  -0.0378504,  -0.0918123,   0.0031022]])        

    def equalish(self,a,b):
        print a
        print b
        dif = gpu.abs(a - b)
        print dif
        self.assertTrue(np.all(dif.as_numpy_array().flatten() < 0.00001))

    def not_equalish(self,a,b):
        dif = gpu.abs(a - b)
        self.assertFalse(np.all(dif.as_numpy_array().flatten() < 0.00001))

    def equalish_atom(self, a, b):
        self.assertTrue(abs(a - b) < 0.00001)



if __name__ == '__main__':
    unittest.main()
