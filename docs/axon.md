# **User Documentation: `array` Class**

The `array` class is designed as a flexible and efficient multidimensional array structure with various functionalities including element-wise operations, mathematical functions, and tensor manipulations. Below are details on how to use this class.

---

## **Class Initialization**

### **Constructor**
```python
array(*data, dtype=None)
```
- **Parameters**:
  - `data`: A list, another `array` object, or a set of integers/floats representing the array's elements.
  - `dtype` (optional): Specifies the data type of the array. Valid types include: `int8`, `int16`, `int32`, `int64`, `float16`, `float32`, `float64`.

- **Example**:
  ```python
  a = array([1, 2, 3, 4], dtype="int32")
  ```

---

## **Array Representation**

### `__repr__` and `__str__`
- These methods allow easy printing and debugging of the array by showing a well-formatted string output with the array's contents.

- **Example**:
  ```python
  a = array([1, 2, 3])
  print(a)  # array([1., 2., 3.], dtype=int32)
  ```

---

## **Array Properties**

### `dtype`
- Returns the data type of the array.

### `shape`
- Returns the shape of the array as a list of dimensions.
  
### `size`
- Returns the total number of elements in the array.

### `ndim`
- Returns the number of dimensions of the array.

### `T`
- Transposed version of the array.

### `F`
- Flattened version of the array.

---

## **Element Access and Manipulation**

### **Indexing and Slicing**
You can access and modify elements using regular Python-style indexing.

- **Get item**:
  ```python
  a = array([[1, 2], [3, 4]])
  print(a[0, 1])  # Output: 2
  ```

- **Set item**:
  ```python
  a[1, 0] = 5
  ```

---

## **Mathematical Operations**

You can perform element-wise operations between arrays or scalars.

### **Addition**
```python
array1 + array2
```

### **Subtraction**
```python
array1 - array2
```

### **Multiplication**
```python
array1 * array2
```

### **Matrix Multiplication**
```python
array1 @ array2
```

### **Power**
```python
array1 ** 2
```

---

## **Mathematical Functions**

The array class supports a wide range of mathematical operations.

### **Common Functions**
- `exp()`: Exponential of each element.
- `log()`: Logarithm of each element.
- `relu()`, `tanh()`, `sigmoid()`: Activation functions used in neural networks.
- `mean(axis=None, keepdims=False)`: Mean of the array elements.
- `sum(axis=None, keepdims=False)`: Sum of the array elements.
- `std(axis=None, ddof=0, keepdims=False)`: Standard deviation of the array elements.

### **Example**:
```python
a = array([1, 2, 3])
a.exp()  # Returns exponential values of the elements
```

---

## **Array Manipulation**

### **Reshape**
```python
a.reshape(new_shape)
```
- **Example**:
  ```python
  a = array([1, 2, 3, 4])
  a = a.reshape((2, 2))  # Reshapes to 2x2
  ```

### **Transpose**
```python
a.T
```

### **Flatten**
```python
a.flatten()
```

### **Unsqueeze and Squeeze**
- `unsqueeze(dim)`: Adds a dimension at the given index.
- `squeeze(dim)`: Removes dimensions of size 1 at the given index.

---

## **Broadcasting**
Broadcasting allows for arrays of different shapes to be compatible for element-wise operations. Use the `broadcast` method to achieve this.

```python
a.broadcast(other)
```

---

## **Utilities**

### **Copying**
You can create a deep copy of the array using the `copy` method.
```python
a_copy = a.copy()
```

### **Conversion**
The `as_type()` method allows for converting the array’s elements to a specified data type.
```python
a.as_type("float32")
```

---

## **Binary Operations**

#### **`__add__(self, other)`**
- **Description**: Adds two arrays element-wise. If shapes are incompatible, it attempts broadcasting.
- **Arguments**: 
  - `other (array or list)`: Array or list to be added.
- **Returns**: A new `array` object with the result of element-wise addition.
- **Raises**: `ValueError` if shapes are incompatible and cannot be broadcasted.

#### **`__mul__(self, other)`**
- **Description**: Multiplies two arrays element-wise. If shapes are incompatible, it attempts broadcasting.
- **Arguments**: 
  - `other (array or list)`: Array or list to be multiplied.
- **Returns**: A new `array` object with the result of element-wise multiplication.
- **Raises**: `ValueError` if shapes are incompatible.

#### **`__matmul__(self, other)`**
- **Description**: Performs matrix multiplication between two arrays.
- **Arguments**:
  - `other (array)`: Another array for matrix multiplication.
- **Returns**: A new `array` object resulting from matrix multiplication.

#### **`__sub__(self, other)`**
- **Description**: Subtracts two arrays element-wise. If shapes are incompatible, it attempts broadcasting.
- **Arguments**:
  - `other (array or list)`: Array or list to be subtracted.
- **Returns**: A new `array` object with the result of element-wise subtraction.
- **Raises**: `ValueError` if shapes are incompatible and cannot be broadcasted.

#### **`__neg__(self)`**
- **Description**: Negates all elements in the array.
- **Arguments**: None.
- **Returns**: A new `array` object with negated values.

#### **`__pow__(self, pow, eps=1e6)`**
- **Description**: Raises every element of the array to the power `pow`. Uses `eps` to handle zero values.
- **Arguments**:
  - `pow (int or float)`: The exponent to which each element will be raised.
  - `eps (float, optional)`: Epsilon value to avoid zero division, default is `1e6`.
- **Returns**: A new `array` object with the result of element-wise exponentiation.

#### **`__truediv__(self, other)`**
- **Description**: Divides two arrays element-wise.
- **Arguments**:
  - `other (array or list)`: The divisor array or list.
- **Returns**: A new `array` object representing the element-wise division.

#### **`__rtruediv__(self, other)`**
- **Description**: Reversed true division for `other / self`.
- **Arguments**:
  - `other (array or list)`: The numerator array or list.
- **Returns**: A new `array` object representing the element-wise reversed division.

---

### **Mathematical Functions**

#### **`exp(self)`**
- **Description**: Applies the exponential function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the result of element-wise exponential calculation.

#### **`log(self)`**
- **Description**: Computes the natural logarithm element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the logarithmic values.
- **Raises**: `ValueError` if any element is non-positive.

#### **`relu(self)`**
- **Description**: Applies the Rectified Linear Unit (ReLU) function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object after applying ReLU activation.

#### **`relu_derivative(self)`**
- **Description**: Computes the derivative of the ReLU function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the derivatives of the ReLU function.

#### **`tanh(self)`**
- **Description**: Applies the hyperbolic tangent function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object after applying the `tanh` activation.

#### **`tanh_derivative(self)`**
- **Description**: Computes the derivative of the hyperbolic tangent function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the derivatives of `tanh`.

#### **`sigmoid(self)`**
- **Description**: Applies the sigmoid activation function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with sigmoid-activated values.

#### **`sigmoid_derivative(self)`**
- **Description**: Computes the derivative of the sigmoid activation function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the derivatives of the sigmoid function.

#### **`gelu(self)`**
- **Description**: Applies the Gaussian Error Linear Unit (GELU) function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object after applying GELU activation.

#### **`gelu_derivative(self)`**
- **Description**: Computes the derivative of the GELU function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the derivatives of the GELU function.

#### **`silu(self)`**
- **Description**: Applies the Sigmoid Linear Unit (SiLU) function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object after applying SiLU activation.

#### **`silu_derivative(self)`**
- **Description**: Computes the derivative of the SiLU function element-wise.
- **Arguments**: None.
- **Returns**: A new `array` object with the derivatives of the SiLU function.

---

### **Statistical Functions**

#### **`mean(self, axis=None, keepdims=False)`**
- **Description**: Computes the mean of the array along the specified axis.
- **Arguments**:
  - `axis (int, optional)`: Axis along which the mean is computed.
  - `keepdims (bool, optional)`: If `True`, retains reduced dimensions as 1.
- **Returns**: A new `array` object with the mean values.

#### **`var(self, axis=None, ddof=0, keepdims=False)`**
- **Description**: Computes the variance of the array along the specified axis.
- **Arguments**:
  - `axis (int, optional)`: Axis along which variance is computed.
  - `ddof (int, optional)`: Delta Degrees of Freedom.
  - `keepdims (bool, optional)`: If `True`, retains reduced dimensions as 1.
- **Returns**: A new `array` object with the variance values.

#### **`std(self, axis=None, ddof=0, keepdims=False)`**
- **Description**: Computes the standard deviation of the array along the specified axis.
- **Arguments**:
  - `axis (int, optional)`: Axis along which standard deviation is computed.
  - `ddof (int, optional)`: Delta Degrees of Freedom.
  - `keepdims (bool, optional)`: If `True`, retains reduced dimensions as 1.
- **Returns**: A new `array` object with the standard deviation values.

#### **`sum(self, axis=None, keepdims=False)`**
- **Description**: Computes the sum of the array along the specified axis.
- **Arguments**:
  - `axis (int, optional)`: Axis along which the sum is computed.
  - `keepdims (bool, optional)`: If `True`, retains reduced dimensions as 1.
- **Returns**: A new `array` object with the summed values.

---

### **Linear Algebra**

#### **`dot(self, other)`**
- **Description**: Computes the dot product of two arrays.
- **Arguments**:
  - `other (array)`: The other array for the dot product.
- **Returns**: A new `array` object with the result of the dot product.

#### **`det(self)`**
- **Description**: Computes the determinant of the array (must be 2D).
- **Arguments**: None.
- **Returns**: A new `array` object with the determinant value.

---