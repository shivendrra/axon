import axon

a = [[1, 4, 5], [1, 4, 5]]
a = axon.array(a, dtype=axon.int64)
c = a @ a.T()
d = c.sum()
print(c)
print(d)