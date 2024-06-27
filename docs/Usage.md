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

#### Numel:
Returns the total number of elements in the array:
```python
a = axon.array([[1, 4, 5], [1, 4, 5]], dtype=axon.int64)
print(a.numel()) # output: 6
```

#### Transpose:
Returns an array that is a transposed version of `input`. The given dims `dim0` and `dim1` are swapped. Two ways to transpose an array:
```python
import axon

a = [[[1, 4, 5], [0, 4, 2]], [[3, 3, -5], [0, -4, 15]]]
a = axon.array(a, dtype=axon.int64)
print(a.T()) # ouptut: array([[[1, 4, 5], [3, 3, -5]], [[0, 4, 2], [0, -4, 15]]], dtype=int64)
print(a.transpose(0, -1)) # output: array([[[1, 0], [4, 4], [5, 2]], [[3, 0], [3, -4], [-5, 15]]], dtype=int64)
```

#### Flatten:
Returns a copy of the array collapsed into one dim. Array can be flatten in two different ways. First is the simple flattening i.e. flatten whole n-dim array into a 1-d array and other is along the axis.

```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[3, 3, -5], [0, -4, 15]]]
a = axon.array(a, dtype=axon.int32)

print(a.F()) # output: [1, 4, 5, 0, 4, 2, 3, 3, -5, 0, -4, 15]
print(a.flatten()) # output: [[1, 4, 5, 0, 4, 2, 3, 3, -5, 0, -4, 15]]
print(a.flatten(1, -1)) # output: [[[1, 4, 5, 0, 4, 2], [3, 3, -5, 0, -4, 15]]]
```

#### Un-squeeze:
Returns a new array with a dim of size one inserted at the specified position.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[3, 3, -5], [0, -4, 15]]]
a = axon.array(a, dtype=axon.int32)

print(a.unsqueeze()) # output: array([[1, 4, 5, 0, 4, 2, 3, 3, -5, 0, -4, 15]], dtype=int32)
print(a.unsqueeze(dim=1)) # output: array([[1, 4, 5, 0, 4, 2], [3, 3, -5, 0, -4, 15]], dtype=int32)
print(a.unsqueeze(dim=-1)) # output: array([[[[1], [4], [5]], [[0], [4], [2]]], [[[3], [3], [-5]], [[0], [-4], [15]]]], dtype=int32)
```

#### Squeeze:
Squeezing removes axes/dim of one from the array.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]]]
a = axon.array(a, dtype=axon.int32)

print(a.squeeze()) # outptut: array([[1, 4, 5], [0, 4, 2]], dtype=int32)
print(a.squeeze(dim=1)) # output: array([[[1, 4, 5], [0, 4, 2]]], dtype=int32)
print(a.squeeze(dim=2)) # output: array([[[1, 4, 5], [0, 4, 2]]], dtype=int32)
```

## Mathematical Operations:

#### Mean:
Returns the mean/array containing means along the axis or in total.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]]
a = axon.array(a, dtype=axon.int32)

print(a.mean()) # output: 2.6666666666666665
print(a.mean(axis=0)) # output: 
print(a.mean(axis=1)) # output: 
```

#### Variance:
Returns the mean/array containing means along the axis or in total.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]]
a = axon.array(a, dtype=axon.int32)

print(a.var()) # output: 3.222222222222222
print(a.var(axis=0))
print(a.var(axis=1)) # output: 
```

#### Standard Deviation:
Returns the mean/array containing means along the axis or in total.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]]
a = axon.array(a, dtype=axon.int32)

print(a.std()) # output: 1.7950549357115013
print(a.std(axis=0)) # output: 
print(a.std(axis=1)) # output: 
```

#### Activations:
Applies different kinds of activation functions on the arrays and returns new array.
```python
import axon
a = [[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]]
a = axon.array(a, dtype=axon.int32)

print(a.tanh()) # output: array([[[0.7615941762924194, 0.9993293285369873, 0.9999092221260071], [0.0, 0.9993293285369873, 0.9640275835990906]], [[0.7615941762924194, 0.9993293285369873, 0.9999092221260071], [0.0, 0.9993293285369873, 0.9640275835990906]]], dtype=float32)
print(a.relu()) # output: array([[[1.0, 4.0, 5.0], [0.0, 4.0, 2.0]], [[1.0, 4.0, 5.0], [0.0, 4.0, 2.0]]], dtype=float32)
print(a.sigmoid()) # output: array([[[0.7310585975646973, 0.9820137619972229, 0.9933071732521057], [0.5, 0.9820137619972229, 0.8807970881462097]], [[0.7310585975646973, 0.9820137619972229, 0.9933071732521057], [0.5, 0.9820137619972229, 0.8807970881462097]]], dtype=float32)
print(a.gelu()) # output: array([[[0.8411920070648193, 3.999929666519165, 5.0], [0.0, 3.999929666519165, 1.9545977115631104]], [[0.8411920070648193, 3.999929666519165, 5.0], [0.0, 3.999929666519165, 1.9545977115631104]]], dtype=float32)
```

## Binary Operations:

#### Addition/Subtraction:
Adds or subtracts two arrays or one array & int and returns a new array.
```python
import axon

a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.int32)
b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

print(a + b) # output: array([[[1, 0, 6], [2, 4, -1]], [[-8, 2, 20], [2, 0, 3]]], dtype=int32)
print(a - b) # output: array([[[1, 8, 4], [-2, 4, 5]], [[10, 6, -10], [-2, 8, 1]]], dtype=int32)
```

#### Multiplication/Division:
Multiplies or divides two arrays or one array & int and returns a new array.
```python
import axon

a = axon.array([[[1, 4, 5], [0, 4, 2]], [[1, 4, 5], [0, 4, 2]]], dtype=axon.int32)
b = axon.array([[[0, -4, 1], [2, 0, -3]], [[-9, -2, 15], [2, -4, 1]]], dtype=axon.int32)

print(a * b) # output: array([[[0, -16, 5], [0, 0, -6]], [[-9, -8, 75], [0, -16, 2]]], dtype=int32)
print(a / b) # output: array([[[0, -1, 5], [0, 0, 0]], [[0, -2, 0], [0, -1, 2]]], dtype=int32)
```