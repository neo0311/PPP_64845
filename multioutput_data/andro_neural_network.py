from pyannet.neural_network import *
from pyannet.data_prep import *

import numpy as np

X_train, y_train, X_test, y_test = tst_train_split(30, "andro.csv", train_test_ratio=0.9,delimiter=',', header_present=False, RandomSeed=True)
andro_nn = NeuralNetwork((30,10,10,6),activations=('swish','swish','linear'))
andro_nn.construct_parameters(initialization=True)
andro_nn.train_and_assess(X_train, y_train,X_test, y_test, type= "SGD", num_of_epochs = 10000, learning_rate = 1e-4 ,stop_condition = 5000, batch_size= 16, optimizer = 'adam', plotting=True, output=True, output_metrics=('rmse', 'mse', 'mae', 'mape', 'r2'))
andro_nn.save_nn('andro')
