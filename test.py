import axon

a = [[1, 4, 5], [1, 4, 5]]
a = axon.array(a, dtype=axon.int64)
print(a.std(axis=0))
print(a.std(axis=1))
print(a.std())
print("----------------")


import numpy as np

print(np.std(a.data, axis=0))
print(np.std(a.data, axis=1))
print(np.std(a.data))