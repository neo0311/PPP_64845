from feamal.neural_network import *
from feamal.data_prep import *
import numpy as np

X_train, y_train, X_test, y_test = test_train_split(4,'dummy.csv')
print(X_train)
print(y_train)
print(X_test)
print(y_test)

print(np.concatenate((X_train, y_train), axis=1))