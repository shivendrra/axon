import axon

a = [[1, 4, 5], [1, 4, 5]]
a = axon.array(a, dtype=axon.int64)
c = a.T() @ a.tanh() ** 2
print(c)