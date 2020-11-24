import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

numDimensions = 2  ##no. of variables to sample from
numDivisions = 10  ##no of divisions at each dimension
numSamples = 10
row = []
for i in range(numDivisions):
     row.append(i)
# LHCube = np.zeros(tuple(LHCubeSize))
Population_indices = np.zeros((numDimensions,numDivisions))
sampleIndices = np.zeros((numDimensions, numSamples))
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
fig, ax = plt.subplots()
ax.scatter(sampleIndices[0,:], sampleIndices[1,:])
plt.grid()
plt.show()
dimensionSpans = [[0,5], [0,5]]
##to solve = last iteration takes k value unnecesserly