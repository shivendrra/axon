# Axon Library Documentation

## Overview
The `axon` library provides a custom implementation of a multidimensional array similar to NumPy arrays, with various utility functions for array manipulation, mathematical operations, and common functional operations. It includes the main `array` class and utility functions for array creation and manipulation.

## Installation
To use the `axon` library, include the `axon` directory in your project and import the necessary modules.

```python
from axon.base import array
from axon.utils import *
```

## Array Class

### Class Definition

```python
class array:
```

### Initialization

```python
def __init__(self, *data: Union[List["array"], list, int, float], dtype: Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']] = None) -> None
```

Initializes an array with the given data and optional data type.

**Parameters:**
- `data`: The data for the array. It can be a list of integers, floats, or other arrays.
- `dtype`: The data type of the array elements (e.g., 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64').

### Representation

```python
def __repr__(self) -> str
```

Returns a string representation of the array.

### Indexing

```python
def __getitem__(self, idx: int)
def __setattr__(self, name: str, value: Any) -> None
def __setitem__(self, index: tuple, value: Any) -> None
def __iter__(self) -> Iterator
```

Supports getting and setting elements using indexing.

### Data Type Conversion

```python
def _convert_dtype(self, data: List["array"], dtype: Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']])
def astype(self, dtype: Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']])
```

Converts the data type of the array elements.

### List Conversion

```python
def tolist(self) -> list
```

Converts the array to a list.

### Copy

```python
def copy(self) -> List["array"]
```

Returns a deep copy of the array.

### View

```python
def view(self, dtype: Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']] = None) -> List["array"]
```

Returns a view of the array with an optional data type.

### Shape and Size

```python
def shape(self) -> list:
def flatten(self) -> list:
def numel(self) -> int:
def size(self) -> tuple:
```

Returns the shape, flattened data, number of elements, and size of the array.

### Transpose and Reshape

```python
def T(self):
def transpose(self, dim0: int, dim1: int):
def reshape(self, new_shape: tuple) -> List["array"]:
```

Transposes and reshapes the array.

### Mathematical Operations

```python
def __add__(self, other: List["array"]) -> List["array"]:
def __mul__(self, other: List["array"]) -> List["array"]:
def __matmul__(self, other: List["array"]) -> List["array"]:
def __pow__(self, exp: float) -> List["array"]:
def __neg__(self) -> List["array"]:
def __sub__(self, other: List["array"]) -> List["array"]:
def __rsub__(self, other: List["array"]) -> List["array"]:
def __rmul__(self, other: List["array"]) -> List["array"]:
def __truediv__(self, other: List["array"]) -> List["array"]:
def rtruediv(self, other: List["array"]) -> List["array"]:
```

Performs element-wise addition, multiplication, matrix multiplication, power, negation, subtraction, and division.

### Summation and Broadcasting

```python
def sum(self, axis: int = None, keepdim: bool = False) -> List["array"]:
def broadcast(self, other: List["array"]) -> List["array"]:
```

Calculates the sum along a specified axis and performs broadcasting with another array.

### Activation Functions

```python
def relu(self) -> List["array"]:
def tanh(self) -> List["array"]:
def sigmoid(self) -> List["array"]:
def gelu(self) -> List["array"]:
```

Applies activation functions (ReLU, tanh, sigmoid, GELU) to the array elements.

## Utility Functions

```python
import random

def zeros(shape)
def ones(shape)
def randint(low, high, size=None, dtype=int)
def arange(start, end, step)
def randn(domain=(1, -1), shape=None)
def zeros_like(arr, dtype=int)
def ones_like(arr, dtype=int)
```

Provides functions for creating arrays with zeros, ones, random integers, ranges, random values, and arrays like other arrays.

### Functions

#### `zeros(shape)`

Creates an array of zeros with the given shape.

**Parameters:**
- `shape`: A tuple specifying the shape of the array.

**Usage:**

```python
zeros((2, 3))
```

#### `ones(shape)`

Creates an array of ones with the given shape.

**Parameters:**
- `shape`: A tuple specifying the shape of the array.

**Usage:**

```python
ones((2, 3))
```

#### `randint(low, high, size=None, dtype=int)`

Generates random integers between `low` and `high`.

**Parameters:**
- `low`: The lower bound of the random integers.
- `high`: The upper bound of the random integers.
- `size`: The number of random integers to generate.
- `dtype`: The data type of the random integers.

**Usage:**

```python
randint(0, 10, size=5)
```

#### `arange(start, end, step)`

Creates an array with values ranging from `start` to `end` with the given `step`.

**Parameters:**
- `start`: The start value.
- `end`: The end value.
- `step`: The step size.

**Usage:**

```python
arange(0, 10, 1)
```

#### `randn(domain=(1, -1), shape=None)`

Generates random values within the specified domain and shape.

**Parameters:**
- `domain`: A tuple specifying the range of random values.
- `shape`: The shape of the array.

**Usage:**

```python
randn(domain=(0, 1), shape=(2, 3))
```

#### `zeros_like(arr, dtype=int)`

Creates an array of zeros with the same shape as the given array.

**Parameters:**
- `arr`: The array to copy the shape from.
- `dtype`: The data type of the new array.

**Usage:**

```python
zeros_like([[1, 2], [3, 4]])
```

#### `ones_like(arr, dtype=int)`

Creates an array of ones with the same shape as the given array.

**Parameters:**
- `arr`: The array to copy the shape from.
- `dtype`: The data type of the new array.

**Usage:**

```python
ones_like([[1, 2], [3, 4]])
```

## Array Class Methods

### Addition

```python
def __add__(self, other: List["array"]) -> List["array"]
```

Performs element-wise addition with another array.

**Parameters:**
- `other`: The other array to add.

**Usage:**

```python
a = array([1, 2, 3])
b = array([4, 5, 6])
c = a + b
print(c)
# Output: array([5, 7, 9], dtype=int64)
```

### Multiplication

```python
def __mul__(self, other: List["array"]) -> List["array"]
```

Performs element-wise multiplication with another array.

**Parameters:**
- `other`: The other array to multiply.

**Usage:**

```python
a = array([1, 2, 3])
b = array([4, 5, 6])
c = a * b
print(c)
# Output: array([4, 10, 18], dtype=int64)
```

### Matrix Multiplication

```python
def __matmul__(self, other: List["array"]) -> List["array"]
```

Performs matrix multiplication with another array.

**Parameters:**
- `other`: The other array to matrix multiply.

**Usage:**

```python
a = array([[1, 2], [3, 4]])
b = array([[5, 6], [7, 8]])
c = a @ b
print(c)
# Output: array([[19, 22], [43, 50]], dtype=int64)
```

### Power

```python
def __pow__(self, exp: float) -> List["array"]
```

Raises each element in the array to the power of `exp`.

**Parameters:**
- `exp`: The exponent.

**Usage:**

```python
a = array([1, 2, 3])
b = a ** 2
print(b)
# Output: array([1, 4, 9], dtype=int64)
```

### Negation

```python
def __neg__(self) -> List["array"]
```

Negates each element in the array.

**Usage:**

```python
a = array([1, 2, 3])
b = -a
print(b)
# Output: array([-1, -2, -3], dtype=int64)
```

### Subtraction

```python
def __sub__(self, other: List["array"]) -> List["array"]
```

Performs element-wise subtraction with another array.

**Parameters:**
- `other`: The other array to subtract.

**Usage:**

```python
a = array([4, 5, 6])
b = array([1, 2, 3])
c = a - b
print(c)
# Output: array([3, 3, 3], dtype=int64)
```

### Right Subtraction

```python
def __rsub__(self, other: List["array"]) -> List["array"]
```

Performs element-wise right subtraction with another array.

**Parameters:**
- `other`: The other array to subtract from.

**Usage:**

```python
a = array([1, 2, 3])
b = array([4, 5, 6])
c = b - a
print(c)
# Output: array([3, 3, 3], dtype=int64)
```

### Right Multiplication

```python
def __rmul__(self, other: List["array"]) -> List["array"]
```

Performs element-wise right multiplication with another array.

**Parameters:**
- `other`: The other array to multiply.

**Usage:**

```python
a = array([1, 2, 3])
b = array([4, 5, 6])
c = b * a
print(c)
# Output: array([4, 10, 18], dtype=int64)
```

### True Division

```python
def __truediv__(self, other: List["array"]) -> List["array"]
```

Performs element-wise division with another array.

**Parameters:**
- `other`: The other array to divide by.

**Usage:**

```python
a = array([4, 6, 8])
b = array([2, 3, 4])
c = a / b
print(c)
# Output: array([2.0, 2.0, 2.0], dtype=int64)
```

### Right True Division

```python
def rtruediv(self, other: List["array"]) -> List["array"]
```

Performs element-wise right division with another array.

**Parameters:**
- `other`: The other array to divide.

**Usage:**

```python
a = array([2, 3, 4])
b = array([4, 6, 8])
c = b / a
print(c)
# Output: array([2.0, 2.0, 2.0], dtype=int64)
```

### Summation

```python
def sum(self, axis: int = None, keepdim: bool = False) -> List["array"]
```

Calculates the sum along a specified axis.

**Parameters:**
- `axis`: The axis to sum along. If None, sums all elements.
- `keepdim`: If True, retains reduced dimensions as dimensions with size one.

**Usage:**

```python
a = array([[1, 2], [3, 4]])
b = a.sum(axis=0)
print(b)
# Output: array([4, 6], dtype=int64)
```

### Broadcasting

```python
def broadcast(self, other: List["array"]) -> List["array"]
```

Broadcasts the given array to the shape of the current array.

**Parameters:**
- `other`: The other array to broadcast.

**Usage:**

```python
a = array([[1, 2], [3, 4]])
b = array([1, 2])
c = a.broadcast(b)
print(c)
# Output: array([[1, 2], [1, 2]], dtype=int64)
```

### ReLU Activation

```python
def relu(self) -> List["array"]
```

Applies the ReLU (Rectified Linear Unit) function element-wise.

**Usage:**

```python
a = array([-1, 2, -3, 4])
b = a.relu()
print(b)
# Output: array([0, 2, 0, 4], dtype=int64)
```

### Tanh Activation

```python
def tanh(self) -> List["array"]
```

Applies the tanh function element-wise.

**Usage:**

```python
a = array([-1, 0, 1])
b = a.tanh()
print(b)
# Output: array([-0.7615941559557649, 0.0, 0.7615941559557649], dtype=int64)
```

### Sigmoid Activation

```python
def sigmoid(self) -> List["array"]
```

Applies the sigmoid function element-wise.

**Usage:**

```python
a = array([-1, 0, 1])
b = a.sigmoid()
print(b)
# Output: array([0.2689414213699951, 0.5, 0.7310585786300049], dtype=int64)
```

### GELU Activation

```python
def gelu(self) -> List["array"]
```

Applies the GELU (Gaussian Error Linear Unit) function element-wise.

**Usage:**

```python
a = array([-1, 0, 1])
b = a.gelu()
print(b)
# Output: array([-0.15865525393145707, 0.0, 0.8413447460685429], dtype=int64)
```

---

This documentation provides detailed descriptions and examples for each method in the `array` class. For more information on the implementation details, please refer to the source code.