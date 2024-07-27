from .base import value
import pickle

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