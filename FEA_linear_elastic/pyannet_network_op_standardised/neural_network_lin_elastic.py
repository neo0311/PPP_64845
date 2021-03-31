from pyannet.neural_network import *
from pyannet.data_prep import *
import numpy as np

X_train, y_train, X_test, y_test = tst_train_split(3, "data_linear_elastic.txt", train_test_ratio=0.8,delimiter=',', header_present=True, RandomSeed=True)
y_train, y_test, y_means, y_stds = data_transform(y_train, y_test)
X_train,X_test, X_means, X_stds = data_transform(X_train,X_test)
linear_elastic_nn = NeuralNetwork((3,8,10,8,6),activations=('swish','swish','swish','linear'))
linear_elastic_nn.construct_parameters(initialization=True)
#linear_elastic_nn.load_nn('linear_elastic_1op_data.npy')
linear_elastic_nn.train_and_assess(X_train, y_train, X_test, y_test, type= "SGD", num_of_epochs = 10000, learning_rate = 1e-4 ,stop_condition = 1000, batch_size= 32, optimizer = 'adam', plotting=True, output=True, output_metrics=('rmse', 'mse', 'mae', 'mape', 'r2'))
linear_elastic_nn.save_nn('linear_elastic_allop_transformed')
#linear_elastic_nn.visualise_test_error(X_test, y_test)