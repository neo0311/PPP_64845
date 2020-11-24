import numpy as np

a = np.asarray(([9,2,3],[4,5,6]))
b = np.ones((2))
print(a)
print(b)
for k in range(len(a)):
    if (b[k] in a[k,:]) == False:
        print('good')