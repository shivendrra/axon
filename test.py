import axon

a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.int32)
b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

print(a.tanh()) # output: 1.7950549357115013
print(a.relu()) # output: 1.7950549357115013
print(a.sigmoid()) # output: 1.7950549357115013
print(a.gelu()) # output: 1.7950549357115013
# print(a.std())
# print(a.std(axis=0))
# print(a.std(axis=1))
# print("----------------")

# print(axon.zeros((2, 5), axon.float16))
# print(axon.ones((2, 5), dtype=axon.float32))
# print(axon.randn(shape=(2, 5), dtype=axon.float16))
# print(axon.zeros_like(a, dtype=axon.float16))
# print(axon.ones_like(a, dtype=axon.int32))

# print("-------------")
# print(axon.zeros((2, 5)))
# print(axon.ones((2, 5)))
# print(axon.randn(shape=(2, 5)))
# print(axon.zeros_like(a))
# print(axon.ones_like(a))