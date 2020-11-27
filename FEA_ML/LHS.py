import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random

numDimensions = 2  ##no. of variables to sample from
numDivisions = 4  ##no of divisions at each dimension
numSamples = 2
dimensionSpans = np.asarray([[50000,400000], [0.2,0.5]])
print(dimensionSpans)
row = []
for i in range(numDivisions):
     row.append(i)
# LHCube = np.zeros(tuple(LHCubeSize))
Population_indices = np.zeros((numDimensions,numDivisions))   #array of all indices
sampleIndices = np.zeros((numDimensions, numSamples), dtype=int)  #array of only sampled indices
for i in range(numDimensions):
    Population_indices[i,:] = np.asarray(row)
Population_indices_temp = Population_indices
#print(Population_indices)
j = 0
while True:
    a = np.zeros((numDimensions))
    for i in range(numDimensions):
        a[i] = np.random.choice(Population_indices_temp[i,:], 1, replace=False)
        #print(a[i], i)
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
                #sampleIndices[k,j] = a[k]
            else:
                temp[k] = a[k]
        #if k == len(sampleIndices)-1:
        if flag ==0:
            sampleIndices[:,j]= temp
            j+=1
    # break
    if j>=numSamples:
        break
print(sampleIndices)
population = np.zeros((numDimensions,numDivisions))     #array for storing sides of hypercube(divisions of each dimension)
eachDivision = np.zeros(numDimensions)
for i in range(numDimensions):
    eachDivision[i] = ((dimensionSpans[i,-1] - dimensionSpans[i,0])/numDivisions)
    for j,value in enumerate(np.arange(dimensionSpans[i,0], dimensionSpans[i,-1], ((dimensionSpans[i,-1] - dimensionSpans[i,0])/numDivisions))):
        #print(i,j)
        population[i,j] = value
print(population)
print('each divisions',eachDivision)
LHCsamples = np.zeros((numDimensions,numSamples))
for i in range(numSamples):
    sampleCoordinates = np.reshape(sampleIndices[:,i],(numDimensions,1))  #array with coordinate (column of sampleIndices) of sampled value
    randomValues = np.reshape(np.random.uniform(low=0, high=eachDivision[0], size=numDimensions), (numDimensions,1))
    #print(randomValues)
    sampleValues = np.take(population, sampleCoordinates) + randomValues
    LHCsamples[:,i] = np.reshape(sampleValues, (2))
    #print(sampleValues)
print(LHCsamples)
fig, ax = plt.subplots()
ax.scatter(LHCsamples[0,:], LHCsamples[1,:])
plt.xticks(population[0,:])
plt.yticks(population[1,:])
plt.yscale('linear')
plt.grid()
plt.show()
##to solve = last iteration takes k value unnecesserly