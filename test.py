import axon


data = [
  [[1, 2], [3, 4], [5, 6]],
  [[7, 8], [9, 10], [11, 12]],
  [[13, 14], [15, 16], [17, 18]]
]

a = axon.array(data, dtype=axon.int8)
print(axon.reshape(data, (2, 9)))
print(axon.reshape(data, (2, 3, 3)))
print(axon.reshape(data, (3, 2, 3)))
print(axon.reshape(data, (2, 2, 3)))