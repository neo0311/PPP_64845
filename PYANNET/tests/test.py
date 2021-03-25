from pyannet.neural_network import *
from pyannet.data_prep import *

from sklearn.metrics import *

y_true = [[0.5, 1], [1, 1], [7, 6]]
y_pred = [[0, 2], [1, 2], [8, 5]]
a = NeuralNetwork((4,2,2), activations=('swish','swish'))

print(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'r2'))
print(r2_score(y_true, y_pred))