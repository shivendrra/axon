import axon

# a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.float16)
# b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

a = axon.array([[1, 2, 3], [4, 5, 6]], dtype=axon.int32)
b = axon.array([[7, 8], [9, 10], [11, 12]], dtype=axon.int32)

print(axon.dot(a, b))