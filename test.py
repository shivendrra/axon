import axon

a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.int32)
b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

# print(a + b)
# print(a - b)
# print(a * b)

print(b + 1)
print(a + [1, 3, 4])

# print(a.var())
# print(a.var(0))
# print(a.var(1))

# print(a.std())
# print(a.std(0))
# print(a.std(1))