import numpy as np

n = 5
start = 0
stop = 4
a = np.arange(start,stop+(stop-start)/n, (stop-start)/n )
print(np.random.uniform(low=1, high=50, size=2))
#print(tuple(a))
a = [1,2 ,3]
b = [4,5,6]
c = np.identity(3)
print(c)
print(np.take(c[1,:], [[1]]))