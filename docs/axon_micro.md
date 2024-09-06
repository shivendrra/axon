# **axon.micro**

## **Class `value`**

The `value` class provides an abstraction for handling numerical values with support for gradients, arithmetic operations, and various activation functions. It acts as a wrapper around `engine.Value` and allows seamless integration of gradient-based computations.

### **Constructor:**
- **`__init__(self, data)`**:
  - **data**: The initial value for this object. It can be a numeric value or an instance of `engine.Value`.

### **Properties:**
- **`data`**:
  - **Getter**: Returns the current value of the `value` object.
  - **Setter**: Updates the value with `new_data`.
- **`grad`**:
  - **Getter**: Returns the gradient of the value.
  - **Setter**: Updates the gradient with `new_data`.

### **Arithmetic Operations:**
- **`__add__(self, other)`**:
  - Adds `self` and `other`. `other` can be a `value` instance or a numeric value.
- **`__radd__(self, other)`**:
  - Handles right addition when `self` is on the right-hand side of the `+` operator.
- **`__mul__(self, other)`**:
  - Multiplies `self` and `other`. `other` can be a `value` instance or a numeric value.
- **`__rmul__(self, other)`**:
  - Handles right multiplication when `self` is on the right-hand side of the `*` operator.
- **`__pow__(self, exp)`**:
  - Raises `self` to the power of `exp`, where `exp` is a numeric value.
- **`__neg__(self)`**:
  - Negates `self`.
- **`__sub__(self, other)`**:
  - Subtracts `other` from `self`. `other` can be a `value` instance or a numeric value.
- **`__rsub__(self, other)`**:
  - Handles right subtraction when `self` is on the right-hand side of the `-` operator.
- **`__truediv__(self, other)`**:
  - Divides `self` by `other`. `other` can be a `value` instance or a numeric value.
- **`__rtruediv__(self, other)`**:
  - Handles right division when `self` is on the right-hand side of the `/` operator.

### **Activation Functions:**
- **`relu(self)`**:
  - Applies the Rectified Linear Unit (ReLU) activation function.
- **`sigmoid(self)`**:
  - Applies the Sigmoid activation function.
- **`tanh(self)`**:
  - Applies the Hyperbolic Tangent (tanh) activation function.
- **`gelu(self)`**:
  - Applies the Gaussian Error Linear Unit (GELU) activation function.
- **`silu(self)`**:
  - Applies the Sigmoid Linear Unit (SiLU) activation function.
- **`swiglu(self)`**:
  - Applies the Swish-Gated Linear Unit (SWiGLU) activation function.

### **Methods:**
- **`backward(self)`**:
  - Performs backpropagation to compute gradients for this value and its dependencies.

### **String Representations:**
- **`__repr__(self)`**:
  - Returns a detailed string representation of the value and its gradient for debugging purposes.
- **`__str__(self)`**:
  - Returns a formatted string representation of the value and its gradient.

### **Example Usage:**

```python
import engine

# Create value instances
a = value(5.0)
b = value(3.0)

# Perform arithmetic operations
c = a + b
d = a * b

# Apply activation functions
relu_result = a.relu()
sigmoid_result = b.sigmoid()

# Perform backward pass
c.backward()

# Print values
print(a)  # Output: axon.micro[Value=5.0000, grad=0.0000]
print(c)  # Output: axon.micro[Value=8.0000, grad=<computed_grad>]
```

# **axon.micro.nn**

## **Module**
The base class for all neural network modules. Provides the basic structure for gradient management, forward execution, and model saving/loading functionality.

### **Methods:**
- **`zero_grad(self)`**:
  - Resets the gradients of all parameters in the module to zero.
- **`parameters(self)`**:
  - Returns a list of all the parameters in the module (to be overridden by subclasses).
- **`forward(self, *inputs, **kwargs)`**:
  - This is a placeholder method that should be implemented by subclasses to define the forward pass.
- **`__call__(self, *inputs, **kwargs)`**:
  - Executes the forward pass using the inputs, mimicking callable behavior.
- **`save(self, filename='model.pickle')`**:
  - Saves the model's state to a file.
- **`load(self, filename='model.pickle')`**:
  - Loads the model's state from a file.

---

## **Neuron**
Represents a single neuron in a neural network layer. It performs a weighted sum of inputs followed by an optional non-linearity.

### **Constructor:**
- **`__init__(self, _in, nonlin=True)`**:
  - **_in**: The number of input features.
  - **nonlin**: A boolean indicating whether to apply a sigmoid non-linearity.

### **Methods:**
- **`__call__(self, x)`**:
  - Performs the forward pass, calculating the weighted sum and applying the non-linearity (sigmoid) if `nonlin` is `True`.
- **`parameters(self)`**:
  - Returns the list of parameters (weights and bias).
- **`__repr__(self)`**:
  - Returns a string representation of the neuron (whether it's a linear or sigmoid neuron).

---

## **Layer**
Represents a fully connected layer consisting of multiple neurons.

### **Constructor:**
- **`__init__(self, n_in, n_out, **kwargs)`**:
  - **n_in**: The number of input features.
  - **n_out**: The number of output features (i.e., the number of neurons in the layer).
  - **kwargs**: Additional parameters passed to each neuron (e.g., nonlin).

### **Methods:**
- **`__call__(self, x)`**:
  - Computes the output of the layer by applying each neuron to the input.
- **`parameters(self)`**:
  - Returns the list of parameters for all neurons in the layer.
- **`__repr__(self)`**:
  - Returns a string representation of the layer, listing all neurons.

---

## **MLP (Multi-Layer Perceptron)**
Represents a multi-layer feedforward neural network (MLP).

### **Constructor:**
- **`__init__(self, n_in, n_out)`**:
  - **n_in**: The number of input features.
  - **n_out**: A list of integers specifying the number of output features for each layer.

### **Methods:**
- **`__call__(self, x)`**:
  - Performs a forward pass through each layer in sequence.
- **`parameters(self)`**:
  - Returns the list of parameters for all layers in the network.
- **`__repr__(self)`**:
  - Returns a string representation of the MLP, listing all layers.

---

## **RNNCell**
Represents a single recurrent neural network (RNN) cell. It processes both the current input and the hidden state to produce an output.

### **Constructor:**
- **`__init__(self, input_size, hidden_size, nonlin=True)`**:
  - **input_size**: The number of input features.
  - **hidden_size**: The number of hidden features.
  - **nonlin**: A boolean indicating whether to apply a ReLU non-linearity to the output.

### **Methods:**
- **`__call__(self, x, h)`**:
  - Takes input `x` and hidden state `h`, processes them, and returns the next hidden state.
  - If `nonlin` is `True`, applies ReLU non-linearity to the output.
- **`parameters(self)`**:
  - Returns the list of parameters for both the input and hidden neurons.
- **`__repr__(self)`**:
  - Returns a string representation of the RNN cell, indicating whether it applies non-linearity or not.

---

## **RNN**
Represents a full recurrent neural network (RNN) with multiple RNN cells stacked in sequence and an output layer.

### **Constructor:**
- **`__init__(self, input_size, hidden_size, output_size, num_layers=1)`**:
  - **input_size**: The number of input features.
  - **hidden_size**: The number of hidden features.
  - **output_size**: The number of output features.
  - **num_layers**: The number of RNN layers (default is 1).

### **Methods:**
- **`__call__(self, x, h=None)`**:
  - Takes input `x` and an optional hidden state `h`, passes them through the RNN cells, and returns the final output.
  - If `h` is not provided, it initializes the hidden state to zero.
- **`parameters(self)`**:
  - Returns the list of parameters for all RNN cells and the output layer.
- **`__repr__(self)`**:
  - Returns a string representation of the RNN, listing all layers and the output layer.

---