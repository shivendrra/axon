# Axon Library

![axonlogo.png](https://github.com/shivendrra/axon/blob/main/logo.png)

You have seen [Micrograd](https://github.com/karpathy/micrograd) by Karpathy, this is the upgraded version of micrograd written in c/c++ & has more functions & operational support. A light weight scalar-level autograd engine written in c/c++ & python

## Installation

Clone the repository:

```bash
git clone https://github.com/shivendrra/axon.git
cd axon
```

## Usage

You can use this similar to micrograd to build a simple neural network or do scalar level backprop.

```python

from axon import value

a = value(2)
b = value(3)

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

To run the unit tests you will have to install PyTorch, which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```shell
python -m pytest
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
