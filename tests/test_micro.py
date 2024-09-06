import torch
from axon.micro import scalar

def test_sanity_check():

  x = scalar(-4.0)
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

  print(ymg.data == ypt.data.item())
  print(xmg.grad == xpt.grad.item())

test_sanity_check()