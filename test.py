import axon

data = [
  [[1, 2], [3, 4], [5, 6]],
  [[7, 8], [9, 10], [11, 12]],
  [[13, 14], [15, 16], [17, 18]]
]

a_1d = [1, 2, 3]
b_1d = [4, 5, 6]

a_2d = [[1, 2, 3], [4, 5, 6]]
b_2d = [[7, 8], [9, 10], [11, 12]]

a, b = axon.array(a_1d), axon.array(b_1d)
print(a.dot(b))
print(axon.dot(a, b))

a, b = axon.array(a_2d), axon.array(b_2d)
print(a.dot(b))
print(axon.dot(a, b))