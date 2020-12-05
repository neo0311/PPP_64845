from feamal.neural_network import *

a = NeuralNetwork((4, 2, 1))
x = np.asarray((-10, -5, 0.0, 5, 10))
print(a.activation(x, type="leakyrelu"))
#print(a.construct_weights())
#print(a.weights_and_biases)
