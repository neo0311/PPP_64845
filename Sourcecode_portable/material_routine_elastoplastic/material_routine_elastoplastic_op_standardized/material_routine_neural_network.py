from neural_network import *
from data_prep import *

import numpy as np

X_train, y_train, X_test, y_test = tst_train_split(7, "data_elasto_plastic.txt", train_test_ratio=0.8,delimiter=',', header_present=True, RandomSeed=True)
X_train,X_test, X_means, X_stds = data_transform(X_train,X_test)
y_train,y_test, y_means, y_stds = data_transform(y_train,y_test)

andro_nn = NeuralNetwork((7,8,9,10,10,14,13,12),activations=('swish','swish','swish','swish','swish','swish','linear'))
#andro_nn.load_nn('material_routine_data.npy')
andro_nn.construct_parameters()
andro_nn.train_and_assess(X_train, y_train,X_test, y_test, type= "SGD", num_of_epochs = 10000, learning_rate = 1e-4 ,stop_condition = 5000, batch_size= 32, optimizer = 'adam', plotting=True, output=True, output_metrics=('rmse', 'mse', 'mae', 'mape', 'r2'))
andro_nn.save_nn('material_routine')
#print(y_test,'\n',andro_nn.predict(X_test))