from feamal.neural_network import *

# a = NeuralNetwork((4, 2, 1))
# b = NeuralNetwork((4, 2, 1))
# x = np.asarray((-10, -5, 0.0, 5, 10))
# #a = [[1], [[1,2],[2,3]]]

# b.construct_weights()
# print(a.weights_and_biases)
# print(b.weights_and_biases)
# a = NeuralNetwork((3,2,2), X = np.ones(3), activations=('relu', 'linear'))
# c = NeuralNetwork((3,2,1), X = np.ones(3), activations=('swish', 'linear'))
# W1 = np.ones((3,2))
# W2 = np.ones((2,2))
# b1 = np.zeros((2))
# b2 = np.zeros((2))
#W = np.asarray((W1, W2),dtype=object)
#b = np.asarray((b1, b2),dtype=object)
#a.construct_weights(method="manual",W = W, b = b)
#c.construct_weights()
#b = a.weights_and_biases['W1']
# print(c.forward_propagate())
# print(c.weights_and_biases)

#print('a')
#print(a.weights_and_biases)
#print('input',a.input, '*', 'W1', W1)
#print('forward', a.forward_propagate())

#print(a.activation(np.asarray((3,3)),'swish'))
#c = np.asarray(b[0])
#print(W[0])
# d = NeuralNetwork((1,1,1), X = np.ones((3,1)), activations=('linear','linear'))
# W1 = 1
# W2 = 1
# b1 = 1
# b2 = 1
# W = np.asarray((W1, W2),dtype=object)
# b = np.asarray((b1, b2),dtype=object)
# d.construct_weights(method='manual', W=W, b=b)
# print(d.forward_propagate())
# m = np.array(([1,1],[1,1],[1,1]))
# print(np.shape(m))
X = np.asarray((1,4,3,0))
a = NeuralNetwork((4,2,3), X , activations=('swish','linear'))
W1 = np.array(([1,1],[0.5,0.25],[0,0.75],[0.25,0.25]))
W2 = np.array(([1,1,1],[1,1,0.5]))
b1 = np.zeros(2)
b2 = np.zeros(3)
W = np.asarray((W1, W2),dtype=object)
b = np.asarray((b1, b2),dtype=object)
a.construct_weights(method='manual', W=W, b=b)

##analytical
input_to_layer_1 = X
input_to_hidden_layer = input_to_layer_1.dot(W1)+ b1
after_activation_of_hidden_layer = a.activation(input_to_hidden_layer, type='swish')
input_to_final_layer = after_activation_of_hidden_layer.dot(W2) + b2
output = a.activation(input_to_final_layer, type='linear')
print(a.forward_propagate())
print(output)