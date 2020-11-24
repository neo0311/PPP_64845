
import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
print(xlimits)
sampling = LHS(xlimits=xlimits)
print(sampling)
num = 500
x = sampling(num)

print(x.shape)

plt.plot(x[:, 0], x[:, 1], "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()