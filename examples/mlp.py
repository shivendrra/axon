import numpy as np

class MLPnp:
  def __init__(self, input_size, hidden_size, output_size):

    self.W1 = np.random.randn(hidden_size, input_size) * 0.01
    self.b1 = np.zeros((hidden_size, 1))
    self.W2 = np.random.randn(output_size, hidden_size) * 0.01
    self.b2 = np.zeros((output_size, 1))
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def sigmoid_derivative(self, z):
    return z * (1 - z)
  
  def forward(self, X):
    self.Z1 = np.dot(self.W1, X) + self.b1
    self.A1 = self.sigmoid(self.Z1)
    self.Z2 = np.dot(self.W2, self.A1) + self.b2
    self.A2 = self.sigmoid(self.Z2)
    return self.A2
  
  def backward(self, X, Y, learning_rate):
    m = X.shape[1]

    # Calculate the gradients
    dZ2 = self.A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
      
    dA1 = np.dot(self.W2.T, dZ2)
    dZ1 = dA1 * self.sigmoid_derivative(self.A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
      
    # Update the weights
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
  
  def train(self, X, Y, iters, learning_rate):
    for i in range(iters):
      output = self.forward(X)
      self.backward(X, Y, learning_rate)
      if i % 100 == 0:
        loss = self.calculate_loss(Y, output)
        print(f"Iteration {i}, loss: {loss}")  
  
  def calculate_loss(self, Y, output):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(output) + (1 - Y) * np.log(1 - output)) / m
    return loss

np.random.seed(42)
X = np.random.randn(2, 100)
Y = (np.sum(X, axis=0) > 0).reshape(1, 100)

mlp = MLPnp(input_size=2, hidden_size=3, output_size=1)
mlp.train(X, Y, iters=1000, learning_rate=0.1)

## MLP written in axon

import axon
import math

class MLP:
  def __init__(self, _in, _out, _hidden) -> None:
    self.wei1 = axon.randn(shape=(_in, _hidden))
    self.b1 = axon.zeros(shape=(1, _hidden))
    self.wei2 = axon.randn(shape=(_hidden, _out))
    self.b2 = axon.zeros(shape=(1, _out))
  
  def forward(self, X):
    self.out1 = X @ self.wei1 + self.b1
    self.out2 = self.sigmoid(self.out1)
    self.out3 = self.out2 @ self.wei2 + self.b2
    self.out4 = self.sigmoid(self.out3)
    return self.out4
  
  def sigmoid(self, z):
    return axon.array([[1 / (1 + math.exp(-val)) for val in row] for row in z])
  
  def sigmoid_derivative(self, z):
    return z * (1 - z)

  def backward(self, X, Y, lr):
    m = X.shape[1]

    dZ2 = self.out4 - Y
    dW2 = (self.wei1 @ dZ2) * (1/m)
    db2 = dZ2.sum(axis=1, keepdim=True) * (1/ m)

    dA1 = (dZ2 @ self.wei2.T())
    dZ1 = dA1 * self.out1.sigmoid_derivative()
    dW1 = dZ1 @ X.T() * (1/m)
    db1 = dZ1.sum(axis=1, keepdim=True) * (1/m)

    self.wei1 -= lr * dW1
    self.b1 -= lr * db1
    self.wei2 -= lr * dW2
    self.b2 -= lr * db2
  
  def train(self, X, Y, iters, lr):
    for i in range(iters):
      output = self.forward(X)
      self.backward(X, Y, lr)
      if i % 100 == 0:
        loss = self.cal_loss(Y, output)
        print(f"iter: {i}, loss: {loss}")
  
  def cal_loss(self, Y, output):
    m = Y.shape[1]
    loss = -((Y * output.log()) + ((1-Y) * (1 - output).log())).sum() / m
    return loss

X = axon.randn(shape=(2, 4))
Y = axon.array([axon.randint(-5, 5, size=10).data for _ in range(1)])
mlp = MLP(4, 10, 2)
out = mlp.forward(X)
mlp.train(X, Y, iters=1000, lr=0.1)