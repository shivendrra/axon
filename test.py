import axon as ax

a = [[1, 4, 5], [1, 4, 5]]
a = ax.array(a, dtype=ax.int8)
c = a + a
print(a)
print(c)