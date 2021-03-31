import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from feamal.neural_network import *
from feamal.data_prep import *
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from keras import backend as K

model = Sequential([
Dense(8, input_dim=3, activation='swish'),
Dense(10, activation='swish'),
Dense(8, activation='swish')])

# def coeff_determination(y_true, y_pred):
#     SS_res =  K.sum(K.square( y_true-y_pred )) 
#     SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#     return ( 1 - SS_res/(SS_tot + K.epsilon()) )

opt = keras.optimizers.SGD(learning_rate=0.0001)
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
model.compile(loss='MSE', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(),"mse", "mae", "mape"])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=128)

# Plot history: RMSE
plt.plot(history.history['root_mean_squared_error'], label='RMSE (training data)')
plt.plot(history.history['val_root_mean_squared_error'], label='RMSE (testing data)')
plt.title('RMSE')
plt.ylabel('rmse value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.grid(axis='both', which='both')
plt.yscale('log')
plt.savefig('keras_plot_assessment_rmse.png')
plt.show()

# Plot history: MSE
plt.plot(history.history['mse'], label='MSE (training data)')
plt.plot(history.history['val_mse'], label='MSE (testing data)')
plt.title('MSE')
plt.ylabel('mse value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.grid(axis='both', which='both')
plt.savefig('keras_plot_assessment_mse.png')
plt.yscale('log')

plt.show()

# Plot history: MAE
plt.plot(history.history['mae'], label='MAE (training data)')
plt.plot(history.history['val_mae'], label='MAE (testing data)')
plt.title('MAE')
plt.ylabel('mae value')
plt.xlabel('No. epoch')
plt.grid(axis='both', which='both')
plt.legend(loc="upper left")
plt.savefig('keras_plot_assessment_mae.png')
plt.yscale('log')

plt.show()

# Plot history: MAPE
plt.plot(history.history['mape'], label='MAPE (training data)')
plt.plot(history.history['val_mape'], label='MAPE (testing data)')
plt.title('MAPE')
plt.ylabel('mape value')
plt.xlabel('No. epoch')
plt.grid(axis='both', which='both')
plt.legend(loc="upper left")
plt.savefig('keras_plot_assessment_mape.png')
plt.yscale('log')

plt.show()