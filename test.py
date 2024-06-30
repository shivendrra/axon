import axon

a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.float16)
b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

print(a * b)