from feamal.neural_network import *
#from sklearn.metrics import mean_squared_error
#import tensorflow as tf
# a = NeuralNetwork((4, 2, 1))
# b = NeuralNetwork((4, 2, 1))
# x = np.asarray((-10, -5, 0.0, 5, 10))
# #a = [[1], [[1,2],[2,3]]]

# b.construct_weights()
# print(a.weights_and_biases)
# print(b.weights_and_biases)
# a = NeuralNetwork((3,2,2), X = np.ones(3), activations=('swish', 'swish'))
# c = NeuralNetwork((3,2,1), X = np.ones(3), activations=('swish', 'swish'))
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
# d = NeuralNetwork((1,1,1), X = np.ones((3,1)), activations=('swish','swish'))
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
# X = np.asarray((1,4,3,0))
# a = NeuralNetwork((4,2,3), X , activations=('swish','swish'))
# W1 = np.array(([1,1],[0.5,0.25],[0,0.75],[0.25,0.25]))
# W2 = np.array(([1,1,1],[1,1,0.5]))
# b1 = np.zeros(2)
# b2 = np.zeros(3)
# W = np.asarray((W1, W2),dtype=object)
# b = np.asarray((b1, b2),dtype=object)
# a.construct_weights(method='manual', W=W, b=b)

# ##analytical
# input_to_layer_1 = X
# input_to_hidden_layer = input_to_layer_1.dot(W1)+ b1
# after_activation_of_hidden_layer = a.activation(input_to_hidden_layer, type='swish')
# input_to_final_layer = after_activation_of_hidden_layer.dot(W2) + b2
# output = a.activation(input_to_final_layer, type='swish')
# print(a.forward_propagate())
# print(output)
# y_pred = np.asarray(([1,0.6],[1,0.7]))
# y = np.asarray(([2,0.3],[2,0.2]))
# y_ = ([1,2,3],([1,2,3]))
# print(np.shape(y_pred))
# print(np.shape(y))
# print(np.shape(y_))


#print(mean_squared_error(y_pred, y, ))
#print(tf.losses.mse(y_pred, y))
a = NeuralNetwork((4,2,1))
x = np.asarray(([0,1,6,-3,6], [0,1,6,-3,6]))
y = a.activation(x,"swish")
z = a.derivatives(x, "swish")
print(x)
print(y)
print(z)
# X = np.asarray((1,4,3,0))
# a = NeuralNetwork((4,2,3), X , activations=('swish','swish'))
# W1 = np.array(([1,1],[0.5,0.25],[0,0.75],[0.25,0.25]))
# W2 = np.array(([1,1,1],[1,1,0.5]))
# b1 = np.zeros(2)
# b2 = np.zeros(3)
# W = np.asarray((W1, W2),dtype=object)
# b = np.asarray((b1, b2),dtype=object)
# a.construct_weights(method='manual', W=W, b=b)
# print(a.derivatives(X,'swish'))
# X1 = np.asarray(([1,4,3,0], [1,4,-3,0]))
# c = NeuralNetwork((4,2,3), X1 , activations=('swish','swish'))
# c.construct_weights(method='manual', W=W, b=b)

# #print((c.forward_propagate()))
# print(c.derivatives(X1,'swish'))