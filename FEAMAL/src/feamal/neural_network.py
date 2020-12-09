import numpy as np


class NeuralNetwork:
    """ A neural network"""
    
    def __init__(self, architecture, X=[], activations=[]):
        """
        architecture: array with neural network structure information
        activations:  array with activation functions to be used in each layer
        """
        self.architecture = np.asarray(architecture ,dtype=int)
        self.activations = np.asarray(activations)
        self.input = X
        self.weights_and_biases = {}
    
    def construct_weights(self, method= "random", W = np.zeros(1), b = np.zeros(1)):
        """Constructs and initializes weights and biases
           random: automatic initialisation with random values
           manual: initialise with given array of weights and biases
                    -takes in an array 'W' with weight matrices and an array 'b' with bias vectors
        """
        #W = np.asarray(W, dtype=object)
        #b = np.asarray(b, dtype=object)

        if method == 'random':
            for i in range(1,len(self.architecture)):
                self.weights_and_biases[f'W{i}'] = np.random.rand(self.architecture[i-1], self.architecture[i])#
                self.weights_and_biases[f'b{i}'] = np.random.rand(self.architecture[i])
        elif method == 'manual':
            for i in range(1,len(self.architecture)):
                self.weights_and_biases[f'W{i}'] = W[i-1]
                self.weights_and_biases[f'b{i}'] = b[i-1] 
        return self.weights_and_biases


    def activation(self, x, type="swish", alpha=0.01):
        """ Defines the activation function"""
        x = np.asarray(x, dtype=float)
        #print(x)
        if type == "swish":
            return x/(1+np.exp(-x))
        
        if type == "linear":
            return x

        if type == "relu":
            return np.maximum(0.0, x)

        if type == "leakyrelu":
            return np.where(x > 0, x, x * alpha)

        if type == "sigmoid":
            return 1/(1+np.exp(-x))
    
    
    def forward_propagate(self):
        """Carries out forward propagation of through the neural network
           returns: the output of the last layer(output)
                 A: ndarray which stores the activated output at each layer
                 Z: variable to temporarly store values before activation
            """
        A = np.zeros((len(self.architecture)), dtype=object)
        A[0] = self.input
        #print(A)
        for layer, activation_function in zip(range(1, len(self.architecture)),self.activations):
            #print(A[layer-1], self.weights_and_biases[f'W{layer}'])
            #print(layer, activation_function)
            Z = (A[layer-1].dot(self.weights_and_biases[f'W{layer}']) + self.weights_and_biases[f'b{layer}'])
            activation_function = self.activations[layer-1]
            A[layer] = self.activation(Z,type=activation_function)
        y_predicted = A[layer]
        #print(A)
        return y_predicted

    def forward_propagate2(self):
        """Carries out forward propagation of through the neural network
           returns: the output of the last layer(output)
                 A: ndarray which stores the activated output at each layer
                 Z: variable to temporarly store values before activation
            """
        A = np.zeros((len(self.architecture)), dtype=object)
        A[0] = self.input
        #print(A)
        for layer, activation_function in zip(range(1, len(self.architecture)),self.activations):
            #print(A[layer-1], self.weights_and_biases[f'W{layer}'])
            #print(layer, activation_function)
            Z = (A[layer-1].dot(self.weights_and_biases[f'W{layer}']) + self.weights_and_biases[f'b{layer}'])
            activation_function = self.activations[layer-1]
            A[layer] = self.activation(Z,type=activation_function)
        y_predicted = A[layer]
        return y_predicted

