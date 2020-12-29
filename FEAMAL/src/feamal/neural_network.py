import numpy as np
from .data_prep import *
import matplotlib.pyplot as plt

class NeuralNetwork:
    """ A neural network"""
    
    def __init__(self, architecture, X=[], y=[], activations=[]):
        """
        architecture: array with neural network structure information
        activations:  array with activation functions to be used in each layer
        """
        self.architecture = np.asarray(architecture ,dtype=int)
        self.activations = np.asarray(activations)
        self.input = X
        self.output = y
        self.weights_and_biases = {}
        self.parameter_gradients ={}
        self.all_data = {}          #empty dictionary for storing derivatives, temporary data, etc
    
    # def construct_weights_back(self, method= "random", W = np.zeros(1), b = np.zeros(1)):
    #     """Constructs and initializes weights and biases
    #        random: automatic initialisation with random values
    #        manual: initialise with given array of weights and biases
    #                 -takes in an array 'W' with weight matrices and an array 'b' with bias vectors
    #     """
    #     #W = np.asarray(W, dtype=object)
    #     #b = np.asarray(b, dtype=object)

    #     if method == 'random':
    #         for i in range(1,len(self.architecture)):
    #             self.weights_and_biases[f'W{i}'] = np.random.rand(self.architecture[i-1], self.architecture[i])#
    #             self.weights_and_biases[f'b{i}'] = np.random.rand(self.architecture[i])
    #     elif method == 'manual':
    #         for i in range(1,len(self.architecture)):
    #             self.weights_and_biases[f'W{i}'] = W[i-1]
    #             self.weights_and_biases[f'b{i}'] = b[i-1] 
    #     return self.weights_and_biases

    def construct_weights(self, method= "random", W = np.zeros(1), b = np.zeros(1)):
        """Constructs and initializes weights and biases
           random: automatic initialisation with random values
           manual: initialise with given array of weights and biases
                    -takes in an array 'W' with weight matrices and an array 'b' with bias vectors
        """
        #W = np.asarray(W, dtype=object)
        #b = np.asarray(b, dtype=object)

        if method == 'random':
            for i in reversed(range(1,len(self.architecture))):
                self.weights_and_biases[f'W{i}'] = np.random.rand(self.architecture[i-1], self.architecture[i])#
                self.weights_and_biases[f'b{i}'] = np.random.rand(self.architecture[i])
        elif method == 'manual':
            for i in reversed(range(len(self.architecture))):
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
    
    # def cost_functions_back(self, y_predicted, y, type="mse"):
    #     """
    #     Defines various cost functions
    #     y_predicted: nD array (mxn) with predicted outputs, m = #datasets, n = #output elements
    #     y          : nD array with actual outputs
    #     mse        : mean squared error
    #     loss       : loss 
    #     """
    #     if y.ndim > 1:
    #         m = np.shape(y)[0]
    #         n = np.shape(y)[1]
    #         mean_over_output_elements = np.sum((y_predicted - y)**2, axis=1)/n
    #         mean_over_all_datasets = np.sum(mean_over_output_elements)/m
    #         loss = mean_over_all_datasets
    #     else: 
    #         mean_over_output_elements = np.sum((y_predicted - y)**2)/len(y)
    #         loss = mean_over_output_elements
    #     return loss

    def cost_functions(self, y_predicted, y, type="mse"):
        """
        Defines various cost functions
        y_predicted: nD array (mxn) with predicted outputs, m = #datasets, n = #output elements
        y          : nD array with actual outputs
        mse        : mean squared error
        loss       : loss 
        """
        if y.ndim > 1:
            m = np.shape(y)[0]  #number of samples
            n = np.shape(y)[1]  #number of output elements
            mean_over_output_elements = np.sum((y_predicted-y)**2, axis=1)/n
            mean_over_all_datasets = np.sum(mean_over_output_elements)/m
            loss = mean_over_all_datasets
        else: 
            mean_over_output_elements = np.sum((y_predicted-y)**2)/(len(y))
            loss = mean_over_output_elements
        return loss

    def forward_propagate_back(self):
        """Carries out forward propagation of through the neural network
           returns: the output of the last layer(output)
                 A: ndarray which stores the activated output at each layer
                 Z: variable to temporarly store values before activation
            """
        A = np.zeros((len(self.architecture)), dtype=object)
        A[0] = self.input
        self.all_data[f'A0'] = A[0]
        for layer, activation_function in zip(range(1, len(self.architecture)),self.activations):
            #print(A[layer-1], self.weights_and_biases[f'W{layer}'])
            #print(layer, activation_function)
            Z = (A[layer-1].dot(self.weights_and_biases[f'W{layer}']) + self.weights_and_biases[f'b{layer}'])
            activation_function = self.activations[layer-1]
            A[layer] = self.activation(Z,type=activation_function)
            self.all_data[f'Z{layer}'] = Z
            self.all_data[f'A{layer}'] = A[layer]
        y_predicted = A[layer]
        return y_predicted



    def forward_propagate(self, X=[]):
        """Carries out forward propagation of through the neural network
           returns: the output of the last layer(output)
                 A: ndarray which stores the activated output at each layer
                 Z: variable to temporarly store values before activation
            """
        A = np.zeros((len(self.architecture)), dtype=object)
        if np.size(X) > 0:
            A[0] = X
        else:
            A[0] = self.input

        self.all_data[f'A0'] = A[0]
        for layer, activation_function in zip(range(1, len(self.architecture)),self.activations):
            #print(A[layer-1], self.weights_and_biases[f'W{layer}'])
            #print(layer, activation_function)
            Z = (A[layer-1].dot(self.weights_and_biases[f'W{layer}']) + self.weights_and_biases[f'b{layer}'])
            activation_function = self.activations[layer-1]
            A[layer] = self.activation(Z,type=activation_function)
            self.all_data[f'Z{layer}'] = Z
            self.all_data[f'A{layer}'] = A[layer]
        y_predicted = A[layer]
        return y_predicted



    def derivatives(self, x, function, alpha=0.01):
        """
        Function with derivatives of the available activation functions
        """
        if function == "sigmoid":
            dadz = self.activation(x,"sigmoid")*(1-self.activation(x,"sigmoid"))
            return dadz

        if function == "swish":
            dadz = self.activation(x,"sigmoid") + x * self.activation(x,"sigmoid") * (1-self.activation(x,"sigmoid"))
            return dadz
        
        if function == "linear":
            dadz = np.ones(np.shape(x))
            return dadz

        if function == "relu":
            dadz = np.greater(x, 0).astype(int)
            return dadz

        if function == "leakyrelu":
            dadz = 1 * (x > 0) + alpha * (x<0)
            return dadz


    def back_propagate_bk(self,y):
        for layer, activation in zip(reversed(range(len(self.architecture))), self.activations[::-1]):
            if layer == len(self.architecture)-1:
                dCda_L = (self.all_data[f'A{layer}'] - y)*(1/len(y))*2
                da_LdZ_L  = self.derivatives(self.all_data[f'Z{layer}'],activation)
                delta_L = np.multiply(dCda_L,da_LdZ_L)
                self.all_data[f'dCda_{layer}'] = dCda_L
                self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_L
                self.all_data[f'delta_{layer}'] = delta_L
            else:
                da_LdZ_l  = self.derivatives(self.all_data[f'Z{layer}'],activation)
                self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_l
                delta_l = np.multiply(np.dot(self.all_data[f'delta_{layer+1}'], (self.weights_and_biases[f'W{layer+1}']).T), da_LdZ_l)                
                self.all_data[f'delta_{layer}'] = delta_l

            dCdW_l = np.outer(self.all_data[f'A{layer-1}'],self.all_data[f'delta_{layer}'])
            dCdb_l = self.all_data[f'delta_{layer}']
            self.all_data[f'dCdW{layer}'] = dCdW_l
            self.all_data[f'dCdb{layer}'] = dCdb_l
            self.parameter_gradients[f'dCdW{layer}'] = dCdW_l
            self.parameter_gradients[f'dCdb{layer}'] = dCdb_l

    def back_propagate(self,y):
        if y.ndim > 1:
            m = np.shape(y)[0]  #number of samples
            n = np.shape(y)[1]  #number of output elements
            for layer, activation in zip(reversed(range(len(self.architecture))), self.activations[::-1]):
                if layer == len(self.architecture)-1:
                    
                    dCda_L = np.sum((self.all_data[f'A{layer}'] - y),axis=0)*(1/(m*n))*2
                    da_LdZ_L  = self.derivatives(np.sum(self.all_data[f'Z{layer}'], axis=0)*(1/m),activation)
                    delta_L = np.multiply(dCda_L,da_LdZ_L)
                    self.all_data[f'dCda_{layer}'] = dCda_L
                    self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_L
                    self.all_data[f'delta_{layer}'] = delta_L
                else:
                    da_LdZ_l  = self.derivatives(np.sum(self.all_data[f'Z{layer}'], axis=0)*(1/m),activation)
                    self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_l
                    delta_l = np.multiply(np.dot(self.all_data[f'delta_{layer+1}'], (self.weights_and_biases[f'W{layer+1}']).T), da_LdZ_l)                
                    self.all_data[f'delta_{layer}'] = delta_l
                
                dCdW_l = np.outer(np.sum(self.all_data[f'A{layer-1}'],axis=0)*(1/m),self.all_data[f'delta_{layer}'])
                dCdb_l = self.all_data[f'delta_{layer}']
                self.all_data[f'dCdW{layer}'] = dCdW_l
                self.all_data[f'dCdb{layer}'] = dCdb_l
                self.parameter_gradients[f'dCdW{layer}'] = dCdW_l
                self.parameter_gradients[f'dCdb{layer}'] = dCdb_l
        else:
            m = 1
            n = len(y)
            for layer, activation in zip(reversed(range(len(self.architecture))), self.activations[::-1]):
                if layer == len(self.architecture)-1:
                    
                    dCda_L = (self.all_data[f'A{layer}'] - y)*(1/(m*n))*2
                    da_LdZ_L  = self.derivatives(self.all_data[f'Z{layer}'],activation)
                    delta_L = np.multiply(dCda_L,da_LdZ_L)
                    self.all_data[f'dCda_{layer}'] = dCda_L
                    self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_L
                    self.all_data[f'delta_{layer}'] = delta_L
                else:
                    da_LdZ_l  = self.derivatives(self.all_data[f'Z{layer}'],activation)
                    self.all_data[f'da_{layer}dZ_{layer}'] = da_LdZ_l
                    delta_l = np.multiply(np.dot(self.all_data[f'delta_{layer+1}'], (self.weights_and_biases[f'W{layer+1}']).T), da_LdZ_l)                
                    self.all_data[f'delta_{layer}'] = delta_l

                dCdW_l = np.outer(self.all_data[f'A{layer-1}'],self.all_data[f'delta_{layer}'])
                dCdb_l = self.all_data[f'delta_{layer}']
                self.all_data[f'dCdW{layer}'] = dCdW_l
                self.all_data[f'dCdb{layer}'] = dCdb_l
                self.parameter_gradients[f'dCdW{layer}'] = dCdW_l
                self.parameter_gradients[f'dCdb{layer}'] = dCdb_l


    def train_nn(self, X_train, y_train, type= "SGD", num_of_epochs = 1000, learning_rate = 0.001,stop_condition = 1000, batch_size= 32, normalisation= 'None', plotting=True, output=True):
        if normalisation == 'z_score':
            X_train = z_score_normalise(X_train)
        elif normalisation == 'None':
            X_train = X_train
        else:
            raise ValueError('Invalid normalisation method')

        if type == "SGD":
            batch_size = 1
        elif type == "BGD":
            batch_size = np.shape(X_train)[0]
        elif type == "miniBGD":
            batch_size = batch_size
        else:
            raise ValueError("undefined method")

        num_of_datasets = np.shape(X_train)[0]
        Generalised_cost = []
        for epoch in range(num_of_epochs):
            individual_loss = []
            

            if type == "SGD":
                X_train, y_train = shuffle_data(X_train, y_train)
                
                for each_sample in range(np.shape(X_train)[0]):
                    y_pred = self.forward_propagate(X_train[each_sample,:])
                    individual_loss.append(self.cost_functions(y_pred, y_train[each_sample,:]))
                    self.back_propagate(y_train[each_sample,:])
                    for layer in range(1,len(self.architecture)):
                        self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * self.parameter_gradients[f'dCdW{layer}']
                        self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * self.parameter_gradients[f'dCdb{layer}']
                
            if type == "BGD":
                y_pred = self.forward_propagate(X_train)
                individual_loss.append(self.cost_functions(y_pred, y_train))
                self.back_propagate(y_train)
                for layer in range(1,len(self.architecture)):
                    self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * self.parameter_gradients[f'dCdW{layer}']
                    self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * self.parameter_gradients[f'dCdb{layer}']

            if type == "miniBGD":
                X_train, y_train = shuffle_data(X_train, y_train)
                for batch in range(0,num_of_datasets+batch_size, batch_size):
                        if batch+batch_size > num_of_datasets:
                            break
                        for each_sample in range(batch+batch_size):
                            y_pred = self.forward_propagate(X_train[each_sample,:])
                            individual_loss.append(self.cost_functions(y_pred, y_train[each_sample,:]))
                            self.back_propagate(y_train[each_sample,:])
                            for layer in range(1,len(self.architecture)):
                                self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * self.parameter_gradients[f'dCdW{layer}']
                                self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * self.parameter_gradients[f'dCdb{layer}']
                    
            if output == True:
                generalised_cost = np.format_float_scientific(np.sum(individual_loss)/len(individual_loss))
                print('Epoch: ', epoch, 'Generalised Cost: ', generalised_cost)
            
            if plotting == True:
                Generalised_cost.append(float(generalised_cost))
            
            if epoch == stop_condition:
                fig, ax = plt.subplots()
                epochs_plot = np.linspace(1,len(Generalised_cost), len(Generalised_cost))
                ax.plot(epochs_plot, Generalised_cost)
                plt.savefig('plot.png')
                plt.show()
                break

    #def test_nn(X_test, y_test)
    
    def predict(self, data):
        return self.forward_propagate(data)




