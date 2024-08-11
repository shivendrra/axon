from axon import array

x = [[1, 3, 5], [2, 5, 6], [2, 5, 6]]
a = array(x, dtype=array.float32)

print(a @ a)

import numpy as np
a = np.array(x)
print(a @ a)