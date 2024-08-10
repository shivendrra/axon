import axon
from axon import array

# a = [[[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
#      [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
#      [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
#      [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]]]

# b = array(a, dtype=axon.float32)

x = [[1, 3, 5], [2, 5, 6], [2, 5, 6]]
a = array(x, dtype=array.float32)

# print("original array: ", b)
# print("sum: ", b.std())
# print("sum1: ", b.std(1))
# print("sum2: ", b.std(2))
# print("sum3: ", b.std(3))

print(a @ a)

import numpy as np
a = np.array(x)
print(a @ a)