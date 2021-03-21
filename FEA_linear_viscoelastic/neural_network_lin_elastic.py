from feamal.neural_network import *
from feamal.data_prep import *

import numpy as np
def dict_to_vector(dictionary):
    
    """
    Converts all the values in a dictionary to a vector
    dictionary: The original dictionary to be converted

    """
    vector = []
    for key in dictionary:
        vector = np.concatenate((vector,dictionary[f'{key}'].flatten()))
    return vector

def vector_to_dict(vector, original_dictionary):
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

X_train, y_train, X_test, y_test = tst_train_split(3, "data_linear_elastic.txt", train_test_ratio=1,delimiter=',', header_present=True, RandomSeed=True)
y_train, y_test, y_means, y_stds = data_transform(y_train, y_test)
X_train,X_test, X_means, X_stds = data_transform(X_train,X_test)
material_routine_nn = NeuralNetwork((3,10, 10,6),activations=('swish','swish','linear'))
material_routine_nn.construct_parameters(initialization=True)
#material_routine_nn.load_nn('material_routine_1op_data.npy')
material_routine_nn.train_nn(X_train, y_train, type= "miniBGD", num_of_epochs = 100000000, learning_rate = 1e-4 ,stop_condition = 50, batch_size= 32, optimizer = 'adam', plotting=True, output=True, output_metrics=('rmse', 'mse', 'mae', 'mape', 'r2'))
material_routine_nn.save_nn('material_routine_allop')
material_routine_nn.visualise_test_error(X_test, y_test)