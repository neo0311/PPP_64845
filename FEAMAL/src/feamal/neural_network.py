import numpy as np
from .data_prep import *
import matplotlib.pyplot as plt

class NeuralNetwork():
    """ A neural network object"""
    
    def __init__(self, architecture, X=[], y=[], activations=[]):
        """
        architecture  :array with neural network structure information
        activations   :array with activation functions to be used in each layer
        X,y           :nd array of input and output datasets(optional)          
        """
        self.architecture = np.asarray(architecture ,dtype=int)
        self.activations = np.asarray(activations)
        self.input = X
        self.output = y
        self.weights_and_biases = {}    #empty array for storing weights and biases
        self.parameter_gradients ={}    #empty array for storing gradients 
        self.all_data = {}              #empty dictionary for storing derivatives, temporary data, etc
    
    def construct_parameters(self, method= "random", W = np.zeros(1), b = np.zeros(1), initialization=True):
        """Constructs and initializes weights and biases
           random          : automatic initialisation with random values
           manual          : initialise with given array of weights and biases
                             -takes in an array 'W' with weight matrices and an array 'b' with bias vectors
           initialization  : 'True' enables Xavier and He initialization methods
        """
        #W = np.asarray(W, dtype=object)
        #b = np.asarray(b, dtype=object)
        for i in reversed(range(1,len(self.architecture))):
            
            if initialization==True:
                if self.activations[i-1] in {'relu' , 'leakyrelu' , 'ealu'}:
                    variance = np.sqrt(2/(self.architecture[i-1]))                          #He initialization
                elif self.activations[i-1] == 'tanh':
                    variance = np.sqrt(6/(self.architecture[i-1] + self.architecture[i]))   #Xavier initialization
                elif self.activations[i-1] in ('swish' , 'sigmoid'):
                    variance = np.sqrt(1/(self.architecture[i-1]))
                else:
                    variance = 1
            
            elif initialization == False:
                variance = 1
            
            if method == 'random':
                    self.weights_and_biases[f'W{i}'] = np.random.rand(self.architecture[i-1], self.architecture[i])*variance #
                    self.weights_and_biases[f'b{i}'] = np.zeros(self.architecture[i])*variance
           
            elif method == 'manual':
                    self.weights_and_biases[f'W{i}'] = W[i-1]
                    self.weights_and_biases[f'b{i}'] = b[i-1] 
        return self.weights_and_biases

    def activation(self, x, type="swish", alpha=0.01):
        """ Defines the activation function
            x    : input array for activation function calculation
            type : defines the activation function to use.
            alpha: a parameter for leakyrelu function
        """

        x = np.asarray(x, dtype=float)
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
    
    def cost_functions(self, y_predicted, y, type="mse"):
        """
        Defines various cost functions
        y_predicted: nD array (mxn) with predicted outputs, m = #datasets, n = #output elements
        y          : nD array with actual outputs
        mse        : mean squared error
        loss       : loss 
        """
        assert(np.shape(y)) == np.shape(y_predicted)
        if type== 'mse':
            if y.ndim > 1:
                m = np.shape(y)[0]  #number of samples
                n = np.shape(y)[1]  #number of output elements
                mean_over_output_elements = np.sum((y_predicted-y)**2, axis=1)/n
                mean_over_all_datasets = np.sum(mean_over_output_elements)/m
                loss = mean_over_all_datasets
            else: 
                mean_over_output_elements = np.sum((y_predicted-y)**2)/(len(y))
                loss = mean_over_output_elements
        else:
            raise ValueError("undefined cost function")
        return loss
    def performance_metric(self, y, y_predicted, type='mse'):

    def forward_propagate(self, X=[]):
        """Carries out forward propagation of through the neural network
           returns: the output of the last layer(output)
                 A: ndarray which stores the activated output at each layer
                 Z: variable to temporarly store input values to activation
            """
        A = np.zeros((len(self.architecture)), dtype=object)
        if np.size(X) > 0:
            A[0] = X
        else:
            A[0] = self.input

        self.all_data[f'A0'] = A[0]
        
        for layer, activation_function in zip(range(1, len(self.architecture)),self.activations):
            Z = (A[layer-1].dot(self.weights_and_biases[f'W{layer}']) + self.weights_and_biases[f'b{layer}'])
            activation_function = self.activations[layer-1]
            A[layer] = self.activation(Z,type=activation_function)
            self.all_data[f'Z{layer}'] = Z
            self.all_data[f'A{layer}'] = A[layer]
        
        y_predicted = A[layer]
        
        return y_predicted

    def derivatives(self, x=[], function='sigmoid', alpha=0.01, y_pred = [], y = []):
        """
        Function with derivatives of the available activation functions and loss functions
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
        
        if function == "mse":
            assert(np.shape(y_pred)) == np.shape(y)
            if y.ndim > 1:
                m = np.shape(y)[0]  #number of samples
                n = np.shape(y)[1]  #number of output elements
                dCdy_pred = np.sum((y_pred - y), axis=0)*(1/(m*n))*2

            else:
                m = 1
                n = len(y) 
                dCdy_pred = (y_pred - y)*(1/(m*n))*2
            return dCdy_pred

    def back_propagate_bk(self,y):
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




    def back_propagate(self,y):
        """
        Function to carry out back propagation. Calculates the gradients of cost with respect to various parameters. Then updates them to a dictionary.
        """
        if y.ndim > 1:
            m = np.shape(y)[0]  #number of samples
            n = np.shape(y)[1]  #number of output elements
            for layer, activation in zip(reversed(range(len(self.architecture))), self.activations[::-1]):
                if layer == len(self.architecture)-1:
                    
                    #dCda_L = np.sum((self.all_data[f'A{layer}'] - y),axis=0)*(1/(m*n))*2
                    dCda_L = self.derivatives(function='mse', y_pred = self.all_data[f'A{layer}'], y= y)
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
                    
                    #dCda_L = (self.all_data[f'A{layer}'] - y)*(1/(m*n))*2
                    dCda_L = self.derivatives(function='mse', y_pred = self.all_data[f'A{layer}'], y=y)
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

    def adam_parameters_update(self,learning_rate=0.001, beta_1= 0.9, beta_2= 0.999, epsilon=10e-8):
        """
        Function to update or initalise the parameters of adam optimizer if needed
        """
        self.adam_parameters = {}
        self.adam_parameters['learning_rate'] = learning_rate
        self.adam_parameters['beta_1'] = beta_1
        self.adam_parameters['beta_2'] = beta_2
        self.adam_parameters['epsilon'] = epsilon


    def train_nn(self, X_train, y_train, type= "SGD", num_of_epochs = 1000, learning_rate = 0.001,stop_condition = 1000, batch_size= 32, optimizer= 'None', plotting=True, output=True):            

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
            
            if optimizer == 'None':
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
                    X_train, y_train = shuffle_data(X_train, y_train)
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
                        y_pred = self.forward_propagate(X_train[batch:batch+batch_size,:])
                        individual_loss.append(self.cost_functions(y_pred, y_train[batch:batch+batch_size,:]))
                        self.back_propagate(y_train[batch:batch+batch_size,:])
                        for layer in range(1,len(self.architecture)):
                            self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * self.parameter_gradients[f'dCdW{layer}']
                            self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * self.parameter_gradients[f'dCdb{layer}']
            elif optimizer == 'adam':
                self.adam_parameters_update()
                self.adam_parameters['learning_rate'] = learning_rate
                beta_1 = self.adam_parameters['beta_1']
                beta_2 = self.adam_parameters['beta_2']
                epsilon = self.adam_parameters['epsilon']

                if type == "SGD":
                    X_train, y_train = shuffle_data(X_train, y_train)
                    iteration = 1
                    for layer in range(1,len(self.architecture)):
                        self.adam_parameters[f'm_w{layer}'] = 0
                        self.adam_parameters[f'm_b{layer}'] = 0
                        self.adam_parameters[f'v_w{layer}'] = 0
                        self.adam_parameters[f'v_b{layer}'] = 0  
                    for each_sample in range(np.shape(X_train)[0]):
                        y_pred = self.forward_propagate(X_train[each_sample,:])
                        individual_loss.append(self.cost_functions(y_pred, y_train[each_sample,:]))
                        self.back_propagate(y_train[each_sample,:])
                        for layer in range(1,len(self.architecture)):
                            #print(layer)
                            #print(self.adam_parameters)
                            self.adam_parameters[f'm_w{layer}'] = beta_1 * self.adam_parameters[f'm_w{layer}'] + (1-beta_1) * self.parameter_gradients[f'dCdW{layer}']
                            self.adam_parameters[f'm_b{layer}'] = beta_1 * self.adam_parameters[f'm_b{layer}'] + (1-beta_1) * self.parameter_gradients[f'dCdb{layer}']
                            self.adam_parameters[f'v_w{layer}'] = beta_2 * self.adam_parameters[f'v_w{layer}'] + (1-beta_2) * self.parameter_gradients[f'dCdW{layer}']**2
                            self.adam_parameters[f'v_b{layer}'] = beta_2 * self.adam_parameters[f'v_b{layer}'] + (1-beta_2) * self.parameter_gradients[f'dCdb{layer}']**2
                                    
                            ##bias correction
                            m_wCorrected = self.adam_parameters[f'm_w{layer}']/(1-beta_1**iteration)
                            m_bCorrected = self.adam_parameters[f'm_b{layer}']/(1-beta_1**iteration)
                            v_wCorrected = self.adam_parameters[f'v_w{layer}']/(1-beta_2**iteration)
                            v_bCorrected = self.adam_parameters[f'v_b{layer}']/(1-beta_2**iteration)
                            ##parameter update
                            self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * (m_wCorrected/(np.sqrt(v_wCorrected) + epsilon))
                            self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * (m_bCorrected/(np.sqrt(v_bCorrected) + epsilon))
                    
                if type == "BGD":
                    X_train, y_train = shuffle_data(X_train, y_train)
                    y_pred = self.forward_propagate(X_train)
                    individual_loss.append(self.cost_functions(y_pred, y_train))
                    self.back_propagate(y_train)
                    for layer in range(1,len(self.architecture)):
                        self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * self.parameter_gradients[f'dCdW{layer}']
                        self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * self.parameter_gradients[f'dCdb{layer}']

                if type == "miniBGD":
                    iteration = 1
                    for layer in range(1,len(self.architecture)):
                        self.adam_parameters[f'm_w{layer}'] = 0
                        self.adam_parameters[f'm_b{layer}'] = 0
                        self.adam_parameters[f'v_w{layer}'] = 0
                        self.adam_parameters[f'v_b{layer}'] = 0
                    #m_w0, m_b0 = [],[]   #moving average of parameter gradients
                    #v_w0, v_b0 = [],[]  
                    X_train, y_train = shuffle_data(X_train, y_train)
                    for batch in range(0,num_of_datasets+batch_size, batch_size):
                        if batch+batch_size > num_of_datasets:
                            break
                        y_pred = self.forward_propagate(X_train[batch:batch+batch_size,:])
                        individual_loss.append(self.cost_functions(y_pred, y_train[batch:batch+batch_size,:]))
                        self.back_propagate(y_train[batch:batch+batch_size,:])
                        for layer in range(1,len(self.architecture)):
                                    #print(layer)
                                    #print(self.adam_parameters)
                            self.adam_parameters[f'm_w{layer}'] = beta_1 * self.adam_parameters[f'm_w{layer}'] + (1-beta_1) * self.parameter_gradients[f'dCdW{layer}']
                            self.adam_parameters[f'm_b{layer}'] = beta_1 * self.adam_parameters[f'm_b{layer}'] + (1-beta_1) * self.parameter_gradients[f'dCdb{layer}']
                            self.adam_parameters[f'v_w{layer}'] = beta_2 * self.adam_parameters[f'v_w{layer}'] + (1-beta_2) * self.parameter_gradients[f'dCdW{layer}']**2
                            self.adam_parameters[f'v_b{layer}'] = beta_2 * self.adam_parameters[f'v_b{layer}'] + (1-beta_2) * self.parameter_gradients[f'dCdb{layer}']**2
                                    
                                    ##bias correction
                            m_wCorrected = self.adam_parameters[f'm_w{layer}']/(1-beta_1**iteration)
                            m_bCorrected = self.adam_parameters[f'm_b{layer}']/(1-beta_1**iteration)
                            v_wCorrected = self.adam_parameters[f'v_w{layer}']/(1-beta_2**iteration)
                            v_bCorrected = self.adam_parameters[f'v_b{layer}']/(1-beta_2**iteration)

                                    ##parameter update
                            self.weights_and_biases[f'W{layer}'] = self.weights_and_biases[f'W{layer}'] - learning_rate * (m_wCorrected/(np.sqrt(v_wCorrected) + epsilon))
                            self.weights_and_biases[f'b{layer}'] = self.weights_and_biases[f'b{layer}'] - learning_rate * (m_bCorrected/(np.sqrt(v_bCorrected) + epsilon))
                        

            if output == True:
                generalised_cost = np.format_float_scientific(np.sum(individual_loss)/len(individual_loss))
                print('Epoch: ', epoch, 'Generalised Cost(mse): ', generalised_cost)
            
            if plotting == True:
                Generalised_cost.append(float(generalised_cost))
            
            if epoch == stop_condition:
                fig, ax = plt.subplots()
                epochs_plot = np.linspace(1,len(Generalised_cost), len(Generalised_cost))
                ax.plot(epochs_plot, Generalised_cost)
                plt.yscale('log')
                plt.savefig('plot.png')
                plt.show()
                break

    def dict_to_vector(self, dictionary):
    
        """
        Converts all the values in a dictionary to a vector
        dictionary: The original dictionary to be converted

        """
        vector = []
        for key in dictionary:
            vector = np.concatenate((vector,dictionary[f'{key}'].flatten()))
        return vector

    def vector_to_dict(self, vector, original_dictionary):
        """
        Converts a vector to a dictionary
        vector   : Vector to be converted
        original_dictionary : Template for conversion
        """
        new_dictionary = {}
        current_index = 0
        for key,item in original_dictionary.items():
            new_index = current_index + item.size
            new_dictionary[f'{key}'] =  np.reshape(vector[current_index:new_index],item.shape)
            current_index = new_index
        return new_dictionary

    def save_nn(self, networkname= 'nn'):
        np.save(f"{networkname}_data.npy", self.weights_and_biases)
        print(f"Data saved to {networkname}_data.npy")

    def load_nn(self, filename):
        self.weights_and_biases = (np.load(filename, allow_pickle=True)).tolist()
        print('Weights and biases are loaded')

    def predict(self, data):
        return self.forward_propagate(data)

    def visualise_test_error(self, X_test, y_test, plotting=True):
        # if data_transformation =='z_score_norm':
        #     X_test = data_transform(X_test, type='z_score_norm')
        # elif data_transformation == 'min_max_norm':
        #     X_test = data_transform(X_test, type='min_max_norm')
        max_error = 0
        for i in range(np.shape(X_test)[0]):
            y_pred = self.forward_propagate(X_test[i,:])
            y = y_test[i,:]
            Loss = self.cost_functions(y_pred,y, type='mse')
            if Loss>max_error:
                max_error = Loss
            print('Relative Error', np.abs(y-y_pred)/np.abs(y))
            #print(X_test[i], y, y_pred)
            print('Test loss(mse): ', Loss)
        #print(max_error)