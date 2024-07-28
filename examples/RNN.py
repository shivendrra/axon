import numpy as np

class SimpleRNN:
  def __init__(self, input_size, hidden_size, output_size):
    self.hidden_size = hidden_size
    
    self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
    self.bh = np.zeros((hidden_size, 1))  # hidden bias
    self.by = np.zeros((output_size, 1))  # output bias
    
  def forward(self, inputs):
    h_prev = np.zeros((self.hidden_size, 1))
    self.last_hs = { -1: h_prev }
    
    for t, x in enumerate(inputs):
      x = x.reshape(-1, 1)
      h_prev = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
      self.last_hs[t] = h_prev
    
    y = np.dot(self.Why, h_prev) + self.by
    return y, h_prev
    
  def backward(self, inputs, targets, learning_rate=0.001):
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    
    dh_next = np.zeros_like(self.last_hs[0])
    
    for t in reversed(range(len(inputs))):
      x = inputs[t].reshape(-1, 1)
      h = self.last_hs[t]
      h_prev = self.last_hs[t-1]
      
      dy = np.copy(self.Why @ h + self.by)
      dy[targets[t]] -= 1  # backprop into y
      
      dWhy += np.dot(dy, h.T)
      dby += dy
      
      dh = np.dot(self.Why.T, dy) + dh_next
      dhraw = (1 - h * h) * dh  # backprop through tanh nonlinearity
      
      dbh += dhraw
      dWxh += np.dot(dhraw, x.T)
      dWhh += np.dot(dhraw, h_prev.T)
      
      dh_next = np.dot(self.Whh.T, dhraw)
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -1, 1, out=dparam)
    
    # Update weights and biases
    self.Wxh -= learning_rate * dWxh
    self.Whh -= learning_rate * dWhh
    self.Why -= learning_rate * dWhy
    self.bh -= learning_rate * dbh
    self.by -= learning_rate * dby

  def train(self, inputs, targets, iters=100, learning_rate=0.001):
    for i in range(iters):
      y, h = self.forward(inputs)
      self.backward(inputs, targets, learning_rate)
      if i % 10 == 0:
        loss = self.calculate_loss(y, targets)
        print(f"Iteration {i}, loss: {loss}")

  def calculate_loss(self, y, targets):
    return np.mean((y - targets) ** 2)

inputs = [np.random.randn(5) for _ in range(10)]
targets = [np.random.randint(0, 2, size=(2, 1)) for _ in range(10)]

rnn = SimpleRNN(input_size=5, hidden_size=10, output_size=2)
rnn.train(inputs, targets, iters=100, learning_rate=0.001)

import axon

class RNN:
  def __init__(self, _in, _hidden, _out) -> None:
    self.hidden_size = _hidden
    self.Wxh = axon.randn(shape=(_hidden, _in))
    self.Whh = axon.randn(shape=(_hidden, _hidden))
    self.Why = axon.randn(shape=(_out, _hidden))
    self.bh = axon.zeros(_hidden, 1)
    self.by = axon.zeros(_out, 1)
  
  def forward(self, input):
    h_prev = axon.zeros((self.hidden_size, 1))
    self.last_hs = { -1: h_prev }
    
    for t, x in enumerate(input):
      x = x.reshape(-1, 1)
      h_prev = ((self.Wxh @ x) + (self.Whh @ h_prev) + self.bh)
      self.last_hs[t] = h_prev
    
    y = (self.Why @ h_prev) + self.by
    return y, h_prev
  
  def backward(self, inputs, targets, lr=0.001):
    dWxh, dWhh, dWhy = axon.zeros_like(self.Wxh), axon.zeros_like(self.Whh), axon.zeros_like(self.Why)
    dbh, dby = axon.zeros_like(self.bh), axon.zeros_like(self.by)

    dh_next = axon.zeros_like(self.last_hs[0])
    for t in reversed(range(len(inputs))):
      x = inputs[t].reshape(-1, 1)
      h = self.last_hs[t]
      h_prev = self.last_hs[t-1]

      dy = ((self.Why @ h) + self.by).copy()
      dy[targets[t]] -= 1

      dWhy += dy @ h.T()
      dby += dy

      dh += (self.Why.T() @ dy) + dh_next
      dhraw = (1 - h * h) * dh

      dbh += dhraw
      dWxh += dhraw @ x.T()
      dWhh += dhraw @ h_prev.T()
      dh_next = self.Whh.T() @ dhraw
    
    for dpram in [dWxh, dWhh, dWhy, dbh, dby]:
      axon.clip(dpram, -1, 1, out=dpram)
    
    self.Wxh -= lr * dWxh
    self.Whh -= lr * dWhh
    self.Why -= lr * dWhy
    self.bh -= lr * dbh
    self.by -= lr * dby
  
  def train(self, inputs, targets, iters=100, lr=0.001):
    for i in range(iters):
      y, h = self.forward(inputs)
      self.backward(inputs, targets, lr)
      if i % 10:
        loss = self.cal_loss(y, targets)
        print(f"iters: {i}, loss: {loss}")
  
  def cal_loss(self, y, targets):
    return axon.mean((y-targets) ** 2)

inputs = axon.randn(shape=(5, 10))
targets = [axon.randint(0, 2, size=(2)) for _ in range(10)]

rnn = RNN(_in=5, _hidden=10, _out=2)
rnn.train(inputs, targets, iters=100, lr=0.001)