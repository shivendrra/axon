from typing import *
from .utils.shape import get_shape, _flatten
from copy import deepcopy
import math

class array:
  def __init__(self, *data, dtype=None) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.dtype = dtype
  
  def __repr__(self) -> str:
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f"array({data_str})"
  
  def __getitem__(self, idx):
    return self.data[idx]
  
  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:int, value: Any):
    if isinstance(index, tuple):
      data = self.data
      grad = self.grad
      for idx in index[:-1]:
        data = data[idx]
        grad = grad[idx]
      data[index[-1]] = value
      grad[index[-1]] = value
    else:
      self.data[index] = value
      self.grad[index] = value

  def __iter__(self) -> Iterator:
    for item in self.data:
      yield item
  
  def tolist(self) -> list:
    return self.data
  
  def copy(self):
    return array(deepcopy(self.data), dtype=self.dtype)
  
  def shape(self) -> list:
    return get_shape(self.data)
  
  def flatten(self) -> list:
    return _flatten(self.data)
  
  def numel(self) -> int:
    out = 1
    for dim in self.shape:
      out = out * dim
    return out
  
  def size(self) -> tuple:
    return tuple(get_shape(self.data))
  
  def __add__(self, other) -> "array":
    other = other if isinstance(other, array) else array(other)
    assert self.shape == other.shape, "shapes are incompatible for operation"

    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b
    
    return array(_add(self.data, other.data), dtype=self.dtype)

  def __mul__(self, other) -> "array":
    other = other if isinstance(other, array) else array(other)
    assert self.shape == other.shape, "shapes are incompatible for operation"

    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    
    return array(_mul(self.data, other.data), dtype=self.dtype)
  
  def __pow__(self, exp) -> "array":
    def _pow(a):
      if isinstance(a, list):
        return [_pow(_a) for _a in a]
      else:
        return math.pow(a, exp)
    return array(_pow(self.data), dtype=self.dtype)

  def __neg__(self) -> "array":
    def _neg(a):
      if isinstance(a, list):
        return [_neg(_a) for _a in a]
      return -a
    return array(_neg(self.data), dtype=self.dtype)

  def __sub__(self, other) -> "array":
    return self + (-other)

  def __rsub__(self, other) -> "array":
    return other + (-self)
  
  def __rmul__(self, other) -> "array":
    return other * self
  
  def __truediv__(self, other) -> "array":
    return self * (other ** -1)
  
  def rtruediv(self, other) -> "array":
    return other * (self ** -1)

  def sum(self, axis=None, keepdim=False) -> "array":
    def _re_sum(data, axis):
      if axis is None:
        return [sum(_flatten(data))]
      elif axis == 0:
        return [sum(row[i] for row in data) for i in range(len(data[0]))]
      else:
        for i in range(len(data[0])):
          for row in data:
            if isinstance(row[i], list):
              return _re_sum(row[i], axis-1)
            return [_re_sum(data[j], None) for j in range(len(data))]

    if axis is not None and (axis < 0 or axis >= len(self.shape)):
      raise ValueError("Axis out of range for the tensor")
    
    out = _re_sum(self.data, axis)
    if keepdim:
      if isinstance(out[0], list):
        out = [item for item in out]
    else:
      out = _flatten(out)
    out = array(out, dtype=self.dtype)
    return out