from feamal.neural_network import *
from feamal.data_prep import *
import numpy
import math
import numpy as np


def test_derivative():
    X = np.asarray(([1,6,3,5], [2,6,2,4]))
    y = np.asarray(([6],[3]))
    a = NeuralNetwork((4,2,1), X,activations= ('swish','linear'))
    a.construct_weights()
    a.forward_propagate()
    a.back_propagate(y)
    print(a.parameter_gradients)
    a.back_propagate_bk(y)
    print(a.parameter_gradients)

test_derivative()