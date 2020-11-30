from feamal.data_prep import LHC
import numpy as np
numDimensions = 2  ##no. of variables to sample from
numDivisions = 10  ##no of divisions at each dimension
numSamples = 9
dimensionSpans = np.asarray([[50000,400000], [0.2,0.5]])
print(LHC(numSamples, numDimensions, numDivisions, dimensionSpans))