
import numpy as np
import matplotlib.pyplot as plt
from feamal.data_prep import *

X_train, y_train, X_test, y_test = test_train_split(6, train_test_ratio=0.8, filename="data_linear_elastic.txt", delimiter=',', header_present=True)
print(X_train)

array= []
header_present = True
delimiter = ','
numInputFeatures= 6
raw_data = np.genfromtxt(filename, delimiter=delimiter, dtype=str)
m = np.shape(raw_data)[0]  #number of lines in file
n = np.shape(raw_data)[1]  #number of columns in file
if header_present == True:
    numTotalData = m-1     #number of total data sets
    firstIndex = 1
else:
    numTotalData = m
    firstIndex = 0
print(numTotalData)
numTrainData, numTestData = int(train_test_ratio*numTotalData), int(numTotalData - int(train_test_ratio*numTotalData)) #number of training and test data
indicesDataSets = (np.linspace(firstIndex,numTotalData-flag,numTotalData)).astype(int) #array with indices of data sets
#print(indicesDataSets)
np.random.shuffle(indicesDataSets)
indicesTrainData = indicesDataSets[:numTrainData]   #indices of train data sets
indicesTestData = indicesDataSets[numTrainData:]   #indices of test data sets
X_train = (np.take(raw_data[:,:numInputFeatures], indicesTrainData, axis=0)).astype(np.float)   #slicing training input data from dataset
y_train = (np.take(raw_data[:,numInputFeatures:], indicesTrainData, axis=0)).astype(np.float)   #slicing training output data from dataset
X_test = (np.take(raw_data[:,:numInputFeatures], indicesTestData, axis=0)).astype(np.float)
y_test = (np.take(raw_data[:,numInputFeatures:], indicesTestData, axis=0)).astype(np.float)
#return X_train, y_train, X_test, y_test