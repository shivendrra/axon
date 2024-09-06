# Axon Library

![axonlogo.png](https://github.com/shivendrra/axon/blob/main/logo.png)

**Axon:** is a lightweight Python library for creating and manipulating multi-dimensional arrays, inspired by libraries such as NumPy. It's written in python only, for now.

**Axon.micro:** You have seen [Micrograd](https://github.com/karpathy/micrograd) by Karpathy, this is the upgraded version of micrograd written in c/c++ & has more functions & operational support. A light weight scalar-level autograd engine written in c/c++ & python

## Features

- Element-wise operations (addition, multiplication, etc.)
- Matrix multiplication
- Broadcasting
- Activation functions (ReLU, tanh, sigmoid, GELU)
- Reshape, transpose, flatten
- Data type conversion
- Micrograd support(Scalar level autograd engine)

## Installation

Clone the repository:

```bash
git clone https://github.com/shivendrra/axon.git
cd axon
```

or

Install via pip:

```bash
pip install axon-pypi
```

## Usage

You can use this similar to micrograd to build a simple neural network or do scalar level backprop.


#### Axon.array

```python
import axon
from axon import array

# Create two 2D arrays
a = array([[1, 2], [3, 4]], dtype=axon.int32)
b = array([[5, 6], [7, 8]], dtype=axon.int32)

# Addition
c = a + b
print("Addition:\n", c)

# Multiplication
d = a * b
print("Multiplication:\n", d)

# Matrix Multiplication
e = a @ b
print("Matrix Multiplication:\n", e)
```

### Output:

```
Addition:
 array([6, 8], [10, 12], dtype=int32)
Multiplication:
 array([5, 12], [21, 32], dtype=int32)
Matrix Multiplication:
 array([19, 22], [43, 50], dtype=int32)
```

anyway, prefer documentation for detailed usage guide:

1. [axon.md](https://github.com/shivendrra/axon/blob/main/docs/axon.md): for development purpose
2. [usage.md](https://github.com/shivendrra/axon/blob/main/docs/usage.md): for using it like numpy
3. [axon_micro.md]((https://github.com/shivendrra/axon/blob/main/docs/axon_micro.md)): for axon.micro i.e. scalar autograd engine

#### Axon.micro
```python

from axon.micro import scalar

a = scalar(2)
b = scalar(3)

c = a + b
d = a * b
e = c.relu()
f = d ** 2.0

f.backward()

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
```

you can even checkout [example](https://github.com/shivendrra/axon/tree/main/examples) neural networks to run them on your system, or build your own :-D.

## Forking the Repository

If you would like to contribute to this project, you can start by forking the repository:

1. Click the "Fork" button at the top right of this page.
2. Clone your forked repository to your local machine:

```bash
git clone https://github.com/shivendrra/axon.git
```

3. Create a new branch:

```bash
git checkout -b my-feature-branch
```

4. Make your changes.
5. Commit and push your changes:

```bash
git add .
git commit -m "Add my feature"
git push origin my-feature-branch
```

6. Create a pull request on the original repository.

## Testing

To run the unit tests you will have to install PyTorch & Numpy, which the tests use as a reference for verifying the correctness of the calculated gradients & calculated values. Then simply run each file according to your prefrence:

```shell
python -m tests/test_array.py # for testing the axon functions with numpy
python -m tests/test_micro.py # for testing the axon.micro functions with pytorch
```

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Ensure all tests pass.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
