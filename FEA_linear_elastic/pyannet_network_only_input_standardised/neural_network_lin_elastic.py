from pyannet.neural_network import *
from pyannet.data_prep import *

import numpy as np


X_train, y_train, X_test, y_test = tst_train_split(3, "data_linear_elastic.txt", train_test_ratio=0.8,delimiter=',', header_present=True, RandomSeed=True)
#y_train, y_test, y_means, y_stds = data_transform(y_train, y_test)
X_train,X_test, X_means, X_stds = data_transform(X_train,X_test)
material_routine_nn = NeuralNetwork((3,10, 10,6),activations=('swish','swish','linear'))
material_routine_nn.construct_parameters(initialization=True)
#material_routine_nn.load_nn('material_routine_1op_data.npy')
material_routine_nn.train_and_assess(X_train, y_train, X_test, y_test, type= "SGD", num_of_epochs = 1000, learning_rate = 1e-4 ,stop_condition =999, batch_size= 32, optimizer = 'adam', plotting=True, output=True, output_metrics=('rmse', 'mse', 'mae', 'mape', 'r2'))
#material_routine_nn.save_nn('material_routine_allop')
#material_routine_nn.visualise_test_error(X_test, y_test)