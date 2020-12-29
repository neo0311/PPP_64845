from feamal.neural_network import *
from feamal.data_prep import *
import numpy as np

X = np.asarray((1,4,3,4))
y = np.asarray((1,4,3))

X_pred = np.asarray((1,4,2, 4))
a = NeuralNetwork((4,2,3), X , activations=('swish','swish'))
W1 = np.array(([1,1],[0.5,0.25],[0,0.75],[0.25,0.25]))
W2 = np.array(([1,1,1],[1,1,0.5]))
b1 = np.zeros(2)
b2 = np.zeros(3)
W = np.asarray((W1, W2),dtype=object)
b = np.asarray((b1, b2),dtype=object)
a.construct_weights(method='manual', W=W, b=b)
#print(a.derivatives(X,'swish'))
X1 = np.asarray(([1,4,3,0], [1,4,2,0], [1,4,3,0]))
X1_pred = np.asarray(([1,4,2,0], [1,4,2,0], [1,4,2,0]))

c = NeuralNetwork((4,2,3), X1 , activations=('swish','linear'))
c.construct_weights(method='manual', W=W, b=b)
a.forward_propagate()
# print(c.derivatives(X1,'swish'))
#print((X1**2))
#print(a.cost_functions(X1,X1_pred))
a.back_propagate(y)
#print(a.weights_and_biases)
dictionary = a.weights_and_biases
#print(a.parameters)
#print(a.weights_and_biases)
#print(np.multiply(np.dot(a.parameters['delta_2'],a.weights_and_biases['W2'].T), a.derivatives(a.parameters['Z1'],'swish')))  

#print('delta_1', np.multiply(first,second))
#print('dCdW1', np.outer(a.parameters['A0'],(np.multiply(first,second))))
#print('dCdW2',a.parameters['dCdW2'])
def dict_to_vector(dictionary):
    vector = []
    for key in dictionary:
        vector = np.concatenate((vector,dictionary[f'{key}'].flatten()))
    return vector

#vector = dict_to_vector(dictionary)

def vector_to_dict(vector, original_dictionary):
    new_dictionary = {}
    current_index = 0
    for key,item in original_dictionary.items():
        new_index = current_index + item.size
        new_dictionary[f'{key}'] =  np.reshape(vector[current_index:new_index],item.shape)
        current_index = new_index
    return new_dictionary

def gradient_checking():
    X = np.asarray([2,3,5,6])
    y = np.asarray([1,6,3])
    a = NeuralNetwork((4,3,3), X , activations=('swish','swish'))
    a.construct_weights()
    #print('original weights', a.weights_and_biases,'\n')
    a.forward_propagate()
    a.back_propagate(y)
    #print('all_data_original', a.all_data, '\n')

    #print('original_gradients', a.parameter_gradients, '\n')
    #print(a.weights_and_biases)
    #print(a.parameters)
    parameters_original = a.weights_and_biases
    original_gradients = a.parameter_gradients
    original_gradients_vector = dict_to_vector(original_gradients)
    parameters_vector = dict_to_vector(parameters_original)
    grad_two_sided_vector = np.zeros(len(parameters_vector))
    epsilon = 1e-7
    for j in range(len(parameters_vector)):
        parameters_vector = dict_to_vector(parameters_original)
        parameters_vector[j] = parameters_vector[j] + epsilon
        a.weights_and_biases = vector_to_dict(parameters_vector,parameters_original)
        #print(a.weights_and_biases, '\n')

        y_pred = a.forward_propagate()
        C_plus = a.cost_functions(y_pred,y) 
        parameters_vector = dict_to_vector(parameters_original)
        parameters_vector[j] = parameters_vector[j] - epsilon
        a.weights_and_biases = vector_to_dict(parameters_vector,parameters_original)
        #print(a.weights_and_biases)

        y_pred = a.forward_propagate()
        C_minus = a.cost_functions(y_pred,y)
        grad_two_sided_vector[j] = (C_plus - C_minus)/(2*epsilon)
        print(j,C_minus-C_plus)
    #print('grad approx', vector_to_dict(grad_two_sided_vector,original_gradients),'\n')
    #print('grad_original', original_gradients,'\n')

    print(grad_two_sided_vector, original_gradients_vector)
    error_difference = (np.linalg.norm(original_gradients_vector-grad_two_sided_vector))/(np.linalg.norm(grad_two_sided_vector) + np.linalg.norm(original_gradients_vector))
    return error_difference
#print(gradient_checking())


def two_layer_network_check():
    X = np.asarray((3))
    a = NeuralNetwork((1,1),X,activations=['swish'])
    a.construct_weights()
    a.forward_propagate()
    print(a.forward_propagate())

def check_complete():
    X = np.asarray((1,2,3))
    y = np.asarray([4,0])
    mine = NeuralNetwork((3,2,2), X, activations=('swish','linear'))
    mine.construct_weights()
    mine.forward_propagate()
    mine.back_propagate(y)
    parameters_vector = dict_to_vector(mine.weights_and_biases)
    gradient_vector = dict_to_vector(mine.parameter_gradients)
    parameters_vector = parameters_vector - 0.01 * gradient_vector
    mine.weights_and_biases = vector_to_dict(parameters_vector, mine.weights_and_biases)
    print(mine.forward_propagate())


def backprop_check():
    eta = 0.8
    X = np.asarray((1,4,3,5))
    a = NeuralNetwork((4,2,10,3), X , activations=('swish','swish','linear'))
    y = np.asarray((5.355,6.7475,10.0945))
    a.construct_weights()
    for i in range(10000):
        y_pred = a.forward_propagate()
        Cost = a.cost_functions(y_pred,y)
        print('iteration', i)
        print('cost', Cost)
        a.back_propagate(y)
        weights_and_biases_vector = dict_to_vector(a.weights_and_biases)
        gradient_vector = dict_to_vector(a.parameter_gradients)
        new_weights_and_biases_vector = weights_and_biases_vector - eta * gradient_vector
        a.weights_and_biases = vector_to_dict(new_weights_and_biases_vector, a.weights_and_biases)
        if Cost<1e-10:
            break

#backprop_check()

def shuffle_check():
    X, y, X_test, y_test = test_train_split(3, 'dummy.txt')
    data = np.concatenate((X, y),axis=1)
    print(X,y)
    np.random.shuffle(data)
    n = np.shape(X)[1]  #num_of_column_elements_first_array
    X_shuffled =   data[:,:n] #slicing input data from dataset
    y_shuffled =    data[:,n:] #slicing output data from dataset
    return X_shuffled, y_shuffled

#X ,y = shuffle_check()
#print(X,y)



def Multiple_inputs():

    X = np.asarray(([1,3],[1,3]))
    y =np.asarray(([1],[1]))
    X_single = np.asarray(([1,3]))
    y_single =np.asarray(([1]))
    a = NeuralNetwork((2,3,1),activations=('swish', 'linear'))
    a.construct_weights()

    y_pred = a.forward_propagate(X)
    a.back_propagate_bk(y)
    print('old method, 2 dtaset',a.parameter_gradients)
    #print('old method, 2 dtaset',a.all_data)
    a.back_propagate(y)
    print('new mwthod, 2 dtaset', a.parameter_gradients)
    #print('new method, 2 dtaset',a.all_data)

    y_pred_single = a.forward_propagate(X_single)




    a.back_propagate_bk(y_single)
    print('old method, single dtaset',a.parameter_gradients)
    #print('old method, single dtaset',a.all_data)

    a.back_propagate(y_single)
    print('new mwthod, sinngle dtaset', a.parameter_gradients)
    #print('new method, single dtaset',a.all_data)


    print('y_pred',y_pred)
    print('y_pred_single',y_pred_single)

    print('cost', a.cost_functions(y_pred,y))
    print('cost_single', a.cost_functions(y_pred_single,y_single))



#Multiple_inputs()

def single_inputs():

    X = np.asarray(([1,3]))
    y =np.asarray(([1]))
    a = NeuralNetwork((2,3,1),activations=('swish', 'linear'))
    a.construct_weights()
    y_pred = a.forward_propagate(X)
    print(y_pred)
    print('cost', a.cost_functions(y_pred,y))

#single_inputs()

def test_linspace():
    n =256
    
    print(np.linspace(1,n,n))


def test_leaky_relu():
