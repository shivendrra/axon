from typing import *
from .utils.shape import get_shape, _flatten, transpose, _re_transpose, broadcasted_shape, broadcasted_array, _reshape
from .dtype import _to_float16, _to_float32, _to_float64, _to_int16, _to_int32, _to_int64, _to_int8
from copy import deepcopy
import math

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'

class array:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64
  float16 = float16
  float32 = float32
  float64 = float64

  def __init__(self, *data, dtype=None) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.dtype = dtype
    if dtype:
      self.data = self._convert_dtype(self.data, dtype)
  
  def __repr__(self) -> str:
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f"array({data_str}, dtype={self.dtype})" if self.dtype is not None else f"array({data_str})"
  
  def __getitem__(self, idx):
    return self.data[idx]
  
  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:int, value: Any) -> None:
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      data[index[-1]] = value
    else:
      self.data[index] = value

  def __iter__(self) -> Iterator:
    for item in self.data:
      yield item

  def _convert_dtype(self, data, dtype):
    def convert(data, dtype):
      if dtype == 'int8':
        return [_to_int8(val) for val in data]
      elif dtype == 'int16':
        return [_to_int16(val) for val in data]
      elif dtype == 'int32':
        return [_to_int32(val) for val in data]
      elif dtype == 'int64':
        return [_to_int64(val) for val in data]
      elif dtype == 'float16':
        return [_to_float16(val) for val in data]
      elif dtype == 'float32':
        return [_to_float32(val) for val in data]
      elif dtype == 'float64':
        return [_to_float64(val) for val in data]
      else:
        raise ValueError("Unsupported dtype")
    
    if isinstance(data, list):
      return [self._convert_dtype(item, dtype) if isinstance(item, list) else convert([item], dtype)[0] for item in data]
    else:
      return convert([data], dtype)[0]
  
  def astype(self, dtype):
    new_data = self._convert_dtype(self.data, dtype)
    return array(new_data, dtype=dtype)
  
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
  
  def T(self):
    return array(transpose(self.data), dtype=self.dtype)
  
  def transpose(self, dim0:int, dim1:int):
    if dim0 >= self.ndim or dim1 >= self.ndim:
      raise ValueError("Transpose dimensions out of range")
    return array(_re_transpose(self.data, dim0, dim1, self.ndim), dtype=self.dtype)

  def reshape(self, new_shape:tuple) -> List["array"]:
    new_shape = new_shape if isinstance(new_shape, tuple) else tuple(new_shape,)
    return array(_reshape(self.data, new_shape), dtype=self.dtype)

  def __add__(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other)
    assert self.shape == other.shape, "shapes are incompatible for operation"

    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b
    
    return array(_add(self.data, other.data), dtype=self.dtype)

  def __mul__(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other)
    assert self.shape == other.shape, "shapes are incompatible for operation"

    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    
    return array(_mul(self.data, other.data), dtype=self.dtype)
  
  def __pow__(self, exp:float) -> List["array"]:
    def _pow(a):
      if isinstance(a, list):
        return [_pow(_a) for _a in a]
      else:
        return math.pow(a, exp)
    return array(_pow(self.data), dtype=self.dtype)

  def __neg__(self) -> List["array"]:
    def _neg(a):
      if isinstance(a, list):
        return [_neg(_a) for _a in a]
      return -a
    return array(_neg(self.data), dtype=self.dtype)

  def __sub__(self, other:List["array"]) -> List["array"]:
    return self + (-other)

  def __rsub__(self, other:List["array"]) -> List["array"]:
    return other + (-self)
  
  def __rmul__(self, other:List["array"]) -> List["array"]:
    return other * self
  
  def __truediv__(self, other:List["array"]) -> List["array"]:
    return self * (other ** -1)
  
  def rtruediv(self, other:List["array"]) -> List["array"]:
    return other * (self ** -1)

  def sum(self, axis:int=None, keepdim:bool=False) -> List["array"]:
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
    return array(out, dtype=self.dtype)
  
  def broadcast(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other)
    new_shape, needs_broadcasting = broadcasted_shape(self.shape, other.shape)
    if needs_broadcasting:
      return array(broadcasted_array(other.data, new_shape), dtype=self.dtype)
    else:
      return None