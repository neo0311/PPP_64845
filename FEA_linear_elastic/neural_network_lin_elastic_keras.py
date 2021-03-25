import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from feamal.neural_network import *
from feamal.data_prep import *
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model = Sequential([
Dense(10, input_dim=3, activation='swish'),BatchNormalization(),
Dense(10, activation='swish')])

model.add(Dense(6, activation='linear'))
X_train, y_train, X_test, y_test = tst_train_split(3, "data_linear_elastic.txt", train_test_ratio=0.8,delimiter=',', header_present=True, RandomSeed=True)
scaler = StandardScaler()
scalar_y = StandardScaler()
scaler.fit(X_train)
scalar_y.fit(y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = scalar_y.transform(y_train)
y_test = scalar_y.transform(y_test)
model.compile(loss='MSE', optimizer='nadam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=128)
# print("Evaluate on test data")
# results = model.evaluate(X_test, y_test, batch_size=128)
# print("test loss, test acc:", results)