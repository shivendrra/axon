# User Documentation

The `array` class provides a flexible and powerful way to handle multi-dimensional arrays, similar to NumPy's `ndarray`. This document serves as a user guide, covering initialization and the various methods available for manipulating arrays.

## Table of Contents

1. [Initialization](#initialization)
2. [Methods](#methods)
   - [as_type](#as_type)
   - [tolist](#tolist)
   - [copy](#copy)
   - [shape](#shape)
   - [numel](#numel)
   - [transpose](#transpose)
   - [flatten](#flatten)
   - [swap_axes](#swap_axes)
   - [unsqueeze](#unsqueeze)
   - [squeeze](#squeeze)
   - [reshape](#reshape)
   - [clip](#clip)
   - [Binary Operations](#binary-operations)
   - [Activation Functions](#activation-functions)
   - [Broadcasting](#broadcast)
   - [Dot Product](#dot)
   - [Determinant](#det)
   - [Mean](#mean)
   - [Variance](#var)
   - [Standard Deviation](#std)
   - [Sum](#sum)

---

## Initialization

### Creating an Array

To initialize an `array` object, you can pass data directly as a list or a set of parameters. Optionally, you can specify the data type (`dtype`).

```python
from your_module import array

# Example: Initialize a 2D array
a = array([[1, 2, 3], [4, 5, 6]], dtype="int32")

# Example: Initialize an array with a single value
b = array(5, dtype="float32")
```

---

## Methods

### as_type

Converts the array to a specified data type.

- **Parameters:**
  - `dtype` (str): The target data type, e.g., `"int8"`, `"float32"`.
  
- **Example:**

  ```python
  c = a.as_type("float64")
  ```

### tolist

Converts the array to a standard Python list.

- **Example:**

  ```python
  list_representation = a.tolist()
  ```

### copy

Creates a deep copy of the array.

- **Example:**

  ```python
  d = a.copy()
  ```

### shape

Returns the shape of the array as a list.

- **Example:**

  ```python
  array_shape = a.shape
  ```

### numel

Returns the total number of elements in the array.

- **Example:**

  ```python
  num_elements = a.numel()
  ```

### transpose

Transposes the array (swaps rows and columns).

- **Example:**

  ```python
  e = a.transpose()
  ```

### flatten

Flattens the array into a 1D list.

- **Parameters:**
  - `start_dim` (int, optional): The first dimension to flatten.
  - `end_dim` (int, optional): The last dimension to flatten.
  
- **Example:**

  ```python
  flat_array = a.flatten()
  ```

### swap_axes

Swaps two axes in the array.

- **Parameters:**
  - `axis1` (int): The first axis.
  - `axis2` (int): The second axis.
  
- **Example:**

  ```python
  swapped_array = a.swap_axes(0, 1)
  ```

### unsqueeze

Adds a new dimension of size 1 at the specified position.

- **Parameters:**
  - `dim` (int, optional): The position where the new dimension should be added.
  
- **Example:**

  ```python
  expanded_array = a.unsqueeze(dim=0)
  ```

### squeeze

Removes dimensions of size 1 from the array.

- **Parameters:**
  - `dim` (int, optional): The dimension to be squeezed.
  
- **Example:**

  ```python
  squeezed_array = a.squeeze(dim=0)
  ```

### reshape

Reshapes the array to a new shape.

- **Parameters:**
  - `new_shape` (tuple): The target shape.
  
- **Example:**

  ```python
  reshaped_array = a.reshape((3, 2))
  ```

### clip

Clips the array values to be within the specified range.

- **Parameters:**
  - `min_value` (float): The minimum value.
  - `max_value` (float): The maximum value.
  
- **Example:**

  ```python
  clipped_array = a.clip(0, 5)
  ```

---

### Binary Operations

#### Addition

Adds two arrays element-wise.

- **Example:**

  ```python
  f = a + b
  ```

#### Multiplication

Multiplies two arrays element-wise.

- **Example:**

  ```python
  g = a * b
  ```

#### Matrix Multiplication

Performs matrix multiplication between two arrays.

- **Example:**

  ```python
  h = a @ b
  ```

#### Subtraction

Subtracts one array from another element-wise.

- **Example:**

  ```python
  i = a - b
  ```

#### Division

Divides one array by another element-wise.

- **Example:**

  ```python
  j = a / b
  ```

### Activation Functions

#### ReLU

Applies the ReLU activation function element-wise.

- **Example:**

  ```python
  relu_array = a.relu()
  ```

#### Tanh

Applies the Tanh activation function element-wise.

- **Example:**

  ```python
  tanh_array = a.tanh()
  ```

#### Sigmoid

Applies the Sigmoid activation function element-wise.

- **Example:**

  ```python
  sigmoid_array = a.sigmoid()
  ```

### Broadcasting

Broadcasts one array to the shape of another.

- **Parameters:**
  - `other` (array): The target array.
  
- **Example:**

  ```python
  broadcasted_array = a.broadcast(b)
  ```

### Dot Product

Computes the dot product of two arrays.

- **Parameters:**
  - `other` (array): The target array.
  
- **Example:**

  ```python
  dot_product = a.dot(b)
  ```

### Determinant

Computes the determinant of the array.

- **Example:**

  ```python
  determinant = a.det()
  ```

### Mean

Computes the mean of the array elements along the specified axis.

- **Parameters:**
  - `axis` (int, optional): The axis along which the mean is computed.
  - `keepdims` (bool, optional): Whether to keep the reduced dimensions.
  
- **Example:**

  ```python
  mean_value = a.mean(axis=0)
  ```

### Variance

Computes the variance of the array elements along the specified axis.

- **Parameters:**
  - `axis` (int, optional): The axis along which the variance is computed.
  - `ddof` (int, optional): Delta degrees of freedom.
  - `keepdims` (bool, optional): Whether to keep the reduced dimensions.
  
- **Example:**

  ```python
  variance_value = a.var(axis=0)
  ```

### Standard Deviation

Computes the standard deviation of the array elements along the specified axis.

- **Parameters:**
  - `axis` (int, optional): The axis along which the standard deviation is computed.
  - `ddof` (int, optional): Delta degrees of freedom.
  - `keepdims` (bool, optional): Whether to keep the reduced dimensions.
  
- **Example:**

  ```python
  std_value = a.std(axis=0)
  ```

### Sum

Computes the sum of the array elements along the specified axis.

- **Parameters:**
  - `axis` (int, optional): The axis along which the sum is computed.
  - `keepdims` (bool, optional): Whether to keep the reduced dimensions.
  
- **Example:**

  ```python
  sum_value = a.sum(axis=0)
  ```