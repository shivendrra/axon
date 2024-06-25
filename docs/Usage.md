# User Documentation

User Guide to use Axon library like NumPy. It has almost all the functions that NumPy has and works in a similar way too, anyway, the guide is here:

## Initializing Array:
Import the axon library or `axon.base` to initialize array.

```python
import axon

a = [[1, 4, 5], [1, 4, 5]]
a = axon.array(a, dtype=axon.int64)
print(a) # output: array([1, 4, 5], [1, 4, 5], dtype=int64)
```

### Class properties & attributes:
Use attributes mentioned below in the same manner to display properties inherited by the array like: `shape`, `size`, `ndim` & `dtype`
```python
print(a.data) # output: [[1, 4, 5], [1, 4, 5]]
print(a.shape) # output: [2, 3]
print(a.size()) # output: (2, 3)
print(a.ndim) # output: 2
print(a.dtype) # output: int64
```

## Unary Operations:
Carry out various unary operations with this library:

#### Sum:
Returns the sum of all elements in the array, axis wise & all total.
```python
import axon

a = axon.array([[1, 4, 5], [1, 4, 5]], dtype=axon.int64)
print(a.sum()) # output: array(20, dtype=int64)
print(a.sum(axis=0)) # output: array([2, 8, 10], dtype=int64)
print(a.sum(axis=1)) # output: array([10, 10], dtype=int64)
```

#### NumEl:
Returns the number of elements in the array:
```python
a = axon.array([[1, 4, 5], [1, 4, 5]], dtype=axon.int64)
print(a.numel()) # output: 6
```

#### Transpose:
Two ways to transpose am array:
```python

import axon
a = [[[1, 4, 5], [0, 4, 2]], [[3, 3, -5], [0, -4, 15]]]
a = axon.array(a, dtype=axon.int64)
print(a.T()) # ouptut: array([[[1, 4, 5], [3, 3, -5]], [[0, 4, 2], [0, -4, 15]]], dtype=int64)
print(a.transpose(0, -1)) # output: array([[[1, 0], [4, 4], [5, 2]], [[3, 0], [3, -4], [-5, 15]]], dtype=int64)
```