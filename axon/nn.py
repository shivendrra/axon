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
    self.w = [value(random.uniform(-1,1)) for _ in range(nin)]
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