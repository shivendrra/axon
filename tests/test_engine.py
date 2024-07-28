import torch
from axon import value

def test_sanity_check():

  x = value(-4.0)
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xmg, ymg = x, y

  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xpt, ypt = x, y

  assert ymg.data == ypt.data.item()
  assert xmg.grad == xpt.grad.item()