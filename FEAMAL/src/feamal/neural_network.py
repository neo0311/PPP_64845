import numpy as np


class NeuralNetwork:
    """ A neural network"""
    
    def __init__(self, architecture):
        """
        architecture: array with neural network architecture information
        """
        self.architecture = np.asarray(architecture ,dtype=int)
        self.weights_and_biases = {}
    
    def construct_weights(self):
        for i in range(1,len(self.architecture)):
            self.weights_and_biases[f'W{i}'] = np.random.rand(self.architecture[i-1], self.architecture[i])#
            self.weights_and_biases[f'b{i}'] = np.random.rand(self.architecture[i])

        return self.weights_and_biases


    def activation(self, x, type="swish", alpha=0.01):
        """ Defines the activation function"""
        x = np.asarray(x)
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
    #def forward_propagate(self):

