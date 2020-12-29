import numpy as np
import matplotlib.pyplot as plt

def LHCSampling(numSamples=int, numDimensions=int, numDivisions=int, dimensionSpans='array', plot=False):
    #print(dimensionSpans)
    row = []
    for i in range(numDivisions):
        row.append(i)
    Population_indices = np.zeros((numDimensions,numDivisions))   #array of all indices
    sampleIndices = np.zeros((numDimensions, numSamples), dtype=int)  #array of only sampled indices
    for i in range(numDimensions):
        Population_indices[i,:] = np.asarray(row)
    Population_indices_temp = Population_indices
    j = 0
    while True:
        a = np.zeros((numDimensions))
        for i in range(numDimensions):
            a[i] = np.random.choice(Population_indices_temp[i,:], 1, replace=False)
        if j == 0:
            sampleIndices[:,j] = a
            j+=1
            continue
        else:
            temp = np.zeros((numDimensions))
            flag = 0
            for k in range(len(sampleIndices)):
                if (a[k] in sampleIndices[k,:j]) == True:
                    flag =1
                    break
                else:
                    temp[k] = a[k]
            if flag ==0:
                sampleIndices[:,j]= temp
                j+=1
        if j>=numSamples:
            break
    #print(sampleIndices)
    population = np.zeros((numDimensions,numDivisions))     #array for storing sides of hypercube(divisions of each dimension)
    eachDivision = np.zeros(numDimensions)    #array for storing each division size for each dimension
    for i in range(numDimensions):
        eachDivision[i] = ((dimensionSpans[i,-1] - dimensionSpans[i,0])/numDivisions)
        for j,value in enumerate(np.arange(dimensionSpans[i,0], dimensionSpans[i,-1], eachDivision[i])):
            population[i,j] = value
    #print('population',population)
    LHCsamples = np.zeros((numDimensions,numSamples))  ##array for storing samples collected using LHC
    sampleValues = np.zeros((numDimensions,1))  ##array for storing a single sample 
    for i in range(numSamples):
        for j in range(numDimensions):
            randomValues = np.random.uniform(low=0, high=eachDivision[j], size=1) #generating a random number for each division for each dimension
            sampleValues[j,0] = np.take(population[j,:], sampleIndices[j,i]) + randomValues  #adding random value with each ticks of a dimension
        #print('random',randomValues)
        LHCsamples[:,i] = np.reshape(sampleValues, (numDimensions))

    if plot==True:
        #fig, ax = plt.subplots()
        plt.scatter(LHCsamples[0,:], LHCsamples[1,:])
        plt.xticks(population[0,:])
        plt.yticks(population[1,:])
        plt.grid()
        plt.show()
    return(LHCsamples)


# numDimensions = 2  ##no. of variables to sample from
# numDivisions = 10  ##no of divisions at each dimension
# numSamples = 9
# dimensionSpans = np.asarray([[50000,400000], [0.2,0.5]])
# print(LHCSampling(numSamples, numDimensions, numDivisions, dimensionSpans))

# def test_train_split_back(numInputFeatures, train_test_ratio=0.8, filename=str, delimiter=',', header_present=True):
#     raw_data = np.genfromtxt(filename, delimiter=delimiter, dtype=str)
#     m = np.shape(raw_data)[0]  #number of lines in file
#     n = np.shape(raw_data)[1]  #number of columns in file
#     if header_present == True:
#         numTotalData = m-1     #number of total data sets
#         firstIndex = 1
#         flag = 0
#     else:
#         numTotalData = m
#         firstIndex = 0
#         flag = 1
#     numTrainData, numTestData = int(train_test_ratio*numTotalData), int(numTotalData - int(train_test_ratio*numTotalData)) #number of training and test data
#     indicesDataSets = (np.linspace(firstIndex,numTotalData-flag,numTotalData)).astype(int) #array with indices of data sets
#     np.random.shuffle(indicesDataSets)
#     indicesTrainData = indicesDataSets[:numTrainData]   #indices of train data sets
#     indicesTestData = indicesDataSets[numTrainData:]   #indices of test data sets
#     X_train = (np.take(raw_data[:,:numInputFeatures], indicesTrainData, axis=0)).astype(np.float)   #slicing training input data from dataset
#     y_train = (np.take(raw_data[:,numInputFeatures:], indicesTrainData, axis=0)).astype(np.float)   #slicing training output data from dataset
#     X_test = (np.take(raw_data[:,:numInputFeatures], indicesTestData, axis=0)).astype(np.float)
#     y_test = (np.take(raw_data[:,numInputFeatures:], indicesTestData, axis=0)).astype(np.float)
#     return X_train, y_train, X_test, y_test

def test_train_split(numInputFeatures, filename_or_array, train_test_ratio=0.8, delimiter=',', header_present=True):
    if type(filename_or_array) == str:
        raw_data = np.genfromtxt(filename_or_array, delimiter=delimiter, dtype=str)
        m = np.shape(raw_data)[0]  #number of lines in file
        n = np.shape(raw_data)[1]  #number of columns in file
        if header_present == True:
            numTotalData = m-1     #number of total data sets
            firstIndex = 1
            flag = 0
        else:
            numTotalData = m
            firstIndex = 0
            flag = 1
    else:
        m = np.shape(filename_or_array)[0]  #number of lines in file
        n = np.shape(filename_or_array)[1]
        numTotalData = m
        firstIndex = 0
        flag = 1
    numTrainData, numTestData = int(train_test_ratio*numTotalData), int(numTotalData - int(train_test_ratio*numTotalData)) #number of training and test data
    indicesDataSets = (np.linspace(firstIndex,numTotalData-flag,numTotalData)).astype(int) #array with indices of data sets
    np.random.shuffle(indicesDataSets)
    indicesTrainData = indicesDataSets[:numTrainData]   #indices of train data sets
    indicesTestData = indicesDataSets[numTrainData:]   #indices of test data sets
    X_train = (np.take(raw_data[:,:numInputFeatures], indicesTrainData, axis=0)).astype(np.float)   #slicing training input data from dataset
    y_train = (np.take(raw_data[:,numInputFeatures:], indicesTrainData, axis=0)).astype(np.float)   #slicing training output data from dataset
    X_test = (np.take(raw_data[:,:numInputFeatures], indicesTestData, axis=0)).astype(np.float)
    y_test = (np.take(raw_data[:,numInputFeatures:], indicesTestData, axis=0)).astype(np.float)
    return X_train, y_train, X_test, y_test
#test_train_split(6, train_test_ratio=0.8, filename="data_linear_elastic.txt", delimiter=',', header_present=True)

def shuffle_data(X, y):
    """
    Shuffles the data sets 
    X : nd array containing the inputs (eg: X_train)
    y : nd array containing the corresponding outputs (eg: y_train)
    returns 
    corresponding shuffled data sets
    """
    data = np.concatenate((X, y),axis=1)
    np.random.shuffle(data)
    n = np.shape(X)[1]  #num_of_column_elements_first_array
    X_shuffled =   data[:,:n] #slicing input data from dataset
    y_shuffled =    data[:,n:] #slicing output data from dataset
    return X_shuffled, y_shuffled

def z_score_normalise(array):
    for i in range(np.shape(array)[1]):
        mean = np.mean(array[:,i])
        std = np.std(array[:,i])
        for j in range(np.shape(array)[0]):
            array[j,i]= (array[j,i] - mean)/std
    return array
