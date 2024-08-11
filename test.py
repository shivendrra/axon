import axon
import math

class MLP:
  def __init__(self, _in, _out, _hidden) -> None:
    self.wei1 = axon.array(axon.randn(shape=(_in, _hidden)), dtype=axon.float32)
    self.b1 = axon.array(axon.zeros(shape=(1, _hidden)), dtype=axon.float32)
    self.wei2 = axon.array(axon.randn(shape=(_hidden, _out)), dtype=axon.float32)
    self.b2 = axon.array(axon.zeros(shape=(1, _out)), dtype=axon.float32)
  
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
    db2 = dZ2.sum(axis=1, keepdims=True) * (1/ m)

    dA1 = (dZ2 @ self.wei2.T)
    dZ1 = dA1 * self.out1.sigmoid_derivative()
    dW1 = dZ1 @ X.T * (1/m)
    db1 = dZ1.sum(axis=1, keepdims=True) * (1/m)

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

X = axon.array(axon.randn(shape=(2, 4)), dtype=axon.float32)
Y = axon.array([axon.randint(-5, 5, size=10) for _ in range(1)], dtype=axon.float32)
mlp = MLP(4, 10, 2)
out = mlp.forward(X)
mlp.train(X, Y, iters=1000, lr=0.1)