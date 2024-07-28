from .base import value
import pickle
import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []
  
  def forward(self, *inputs, **kwargs):
    raise NotImplementedError('forward not written')

  def __call__(self, *inputs, **kwargs):
    return self.forward(*inputs, **kwargs)
  
  def save(self, filename='model.pickle'):
    with open(filename, 'wb') as f:
      pickle.dump(self.save_dict(), f)

  def load(self, filename='model.pickle'):
    with open(filename, 'rb') as f:
      state = pickle.load(f)
    self.load_dict(state)

class Neuron(Module):
  def __init__(self, nin, nonlin=True) -> None:
    super().__init__()
    self.w = [value(random.uniform(-0.5,0.5)) for _ in range(nin)]
    self.b = value(0)
    self.nonlin = nonlin
  
  def __call__(self, x):
    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
  def __init__(self, _in, _out, **kwargs) -> None:
    super().__init__()
    self.neurons = [Neuron(_in, **kwargs) for _ in range(_out)]
  
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self) -> str:
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class RNNCell(Module):
  def __init__(self, input_size, hidden_size, nonlin=True):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.wx = [value(random.uniform(-0.5, 0.5)) for _ in range(input_size * hidden_size)]
    self.wh = [value(random.uniform(-0.5, 0.5)) for _ in range(hidden_size * hidden_size)]
    self.b = [value(0) for _ in range(hidden_size)]
    self.nonlin = nonlin

  def __call__(self, x, h):
    wx = sum((self.wx[i] * x[i] for i in range(self.input_size)), 0)
    wh = sum((self.wh[i] * h[i] for i in range(self.hidden_size)), 0)
    act = wx + wh + self.b
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.wx + self.wh + self.b

  def __repr__(self):
    return f"{'ReLU' if self.nonlin else 'Linear'}RNNCell({self.input_size}, {self.hidden_size})"


class RNN(Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn_cells = [RNNCell(input_size, hidden_size) if i == 0 else RNNCell(hidden_size, hidden_size)
                      for i in range(num_layers)]
    self.output_layer = Layer(hidden_size, output_size)

  def __call__(self, x, h=None):
    if h is None:
        h = [value(0) for _ in range(self.hidden_size)]
    for rnn_cell in self.rnn_cells:
        h = rnn_cell(x, h)
    return self.output_layer(h)

  def parameters(self):
    return [p for rnn_cell in self.rnn_cells for p in rnn_cell.parameters()] + self.output_layer.parameters()

  def __repr__(self):
    return f"RNN of [{', '.join(str(rnn_cell) for rnn_cell in self.rnn_cells)}, {self.output_layer}]"