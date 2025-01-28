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
from axon import array

class MLP:
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    self.w1 = array(axon.random.randn(_hid, _in), dtype=axon.float32)
    self.b1 = array(axon.zeros(_hid, 1))
    self.w2 = array(axon.random.randn(_out, _hid), dtype=axon.float32)
    self.b2 = array(axon.zeros(_out, 1))
  
  def forward(self, x):
    self.Z1 = axon.dot(self.w1, x) + self.b1
    self.A1 = self.Z1.sigmoid()
    self.Z2 = axon.dot(self.w2, self.A1) + self.b2
    self.A2 = self.Z2.sigmoid()
    return self.A2
  
  def backward(self, X, Y, lr):
    m = X.shape[1]

    dz2 = self.A2 - Y
    dw2 = axon.dot(dz2, self.A1.T) * (1 / m)
    db2 = axon.sum(dz2, axis=1, keepdims=True) * (1 / m)
    
    da1 = axon.dot(self.w2.T, dz2)
    print(da1, da1.shape)
    print(self.A1, self.A1.shape)
    dz1 = da1 * self.A1.sigmoid_derivative()
    dw1 = axon.dot(dz1, X.T) * (1 / m)
    db1 = axon.sum(dz1, axis=1, keepdims=True) * (1 / m)

    self.w1 = self.w1 + (dw1 * -lr)
    # self.b1 = self.b1 + (db1 * -lr)
    self.w2 = self.w2 + (dw2 * -lr)
    # self.b2 = self.b2 + (db2 * -lr)
  
  def train(self, X, Y, iters, lr):
    for i in range(iters):
      output = self.forward(X)
      self.backward(X, Y, lr)
      if i % 100 == 0:
        loss = self.calculate_loss(Y, output)
        print(f"Iter: {i}, loss: {loss.data[0]:.4f}")
    
  def calculate_loss(self, Y, output):
    m = Y.shape[1]
    loss = ((Y - output) ** 2 / m).sum()
    return loss

X = array(axon.random.randn(2, 100), dtype='float32')
Y = (axon.sum(X, axis=0).reshape((1, 100)))

mlp = MLP(2, 10, 1)
mlp.train(X, Y, iters=2000, lr=0.01)