from feamal.neural_network import *
# Point.__eq__
def test_nn_correctly_takes_architecture_data():
    assert(NeuralNetwork((4, 2, 1)).architecture.all() == np.asarray((4, 2, 1)).all()) == True

def test_nn_activation_swish():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-20, -1.0, 0.0, 1.0, 20)],type="swish")).all() == (np.asarray((-4.12230724e-08,-2.68941421e-01,0.00000000e+00,7.31058579e-01, 2.00000000e+01))).all()

def test_nn_activation_sigmoid():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-20, -1.0, 0.0, 1.0, 20)],type="sigmoid")).all() == (np.asarray((2.06115362e-09, 2.68941421e-01, 5.00000000e-01, 7.31058579e-01, 9.99999998e-01))).all()

def test_nn_activation_relu():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray(( 0.,  0., 0. , 5., 10.))).all()

def test_nn_activation_linear():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray((-10, -5, 0.0, 5, 10))).all()

def test_nn_activation_leakyrelu():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray((-0.1, -0.05,  0.,    5.,   10.  ))).all()

