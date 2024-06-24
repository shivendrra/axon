import axon

a = [[1, 4, 5], [1, 4, 5]]
a = axon.array(a, dtype=axon.int64)
print(a.std(axis=0))
print(a.std(axis=1))
print(a.std())
print("----------------")

print(axon.zeros((2, 5), axon.float16))
print(axon.ones((2, 5), dtype=axon.float32))
print(axon.randn(shape=(2, 5), dtype=axon.float16))
print(axon.zeros_like(a, dtype=axon.float16))
print(axon.ones_like(a, dtype=axon.int32))

print("-------------")
print(axon.zeros((2, 5)))
print(axon.ones((2, 5)))
print(axon.randn(shape=(2, 5)))
print(axon.zeros_like(a))
print(axon.ones_like(a))