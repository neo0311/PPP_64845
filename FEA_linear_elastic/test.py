import numpy as np

u = np.ones((1,2))
v = np.ones((2,1))
for i in range(10):
    A = np.random.randint(10, size=(2, 2))
    print(A, u@A@v, (u@A@v)/v@u)
