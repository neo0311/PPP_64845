from feamal.data_prep import *
dimensionSpans = np.asarray([[45000,450000], [0.2,0.5]])
Samples = LHCSampling(40,2,100, dimensionSpans,plot=True)
for E,v in zip(Samples[0,:],Samples[1,:]):
    print(E,v)