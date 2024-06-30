from typing import *
from .helpers.utils import _zeros
from .helpers.shape import get_shape, _flatten, broadcasted_shape, broadcasted_array, reshape, re_flat, _unsqueeze, _squeeze, mean_axis, var_axis
from .helpers.functionals import tanh, sigmoid, gelu, relu
from .dtypes.dtype import *
from .dtypes.convert import handle_conversion
from copy import deepcopy
import math

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
long = 'long'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
double = 'double'

def handle_float(data1, data2):
  dtype1, dtype2 = data1.dtype, data2.dtype
  if dtype1 == array.float16 or dtype2 == array.float16:
    return True
  else:
    return False

class array:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64 or long
  float16 = float16
  float32 = float32
  float64 = float64 or double

  def __init__(self, *data:Union[List["array"], list, int, float], dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.dtype = array.int32 if dtype is None else dtype
    if dtype is not None:
      self.data = self._convert_dtype(self.data, dtype)
  
  def __repr__(self) -> str:
    if self.ndim == 1:
      return f"array([{self.data}], dtype={self.dtype})"
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f"array([{data_str}], dtype={self.dtype})"
  
  def __getitem__(self, index:tuple):
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      return data[index[-1]]
    else:
      return self.data[index]
  
  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:tuple, value: Any) -> None:
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

  def _convert_dtype(self, data:List["array"], dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]):
    return handle_conversion(data, dtype)
  
  def astype(self, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]) -> List["array"]:
    new_data = handle_conversion(self.data, dtype)
    return array(new_data, dtype=dtype)
  
  def tolist(self) -> list:
    return self.data
  
  def copy(self) -> List["array"]:
    return array(deepcopy(self.data), dtype=self.dtype)
  
  def view(self, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> List["array"]:
    new_array = array(self.data)
    if dtype is not None:
      new_array.data = self._convert_dtype(new_array.data, dtype)
      new_array.dtype = dtype
    return new_array
  
  def shape(self) -> list:
    return get_shape(self.data)
  
  def flatten(self, start_dim:int=0, end_dim:int=-1) -> list:
    return re_flat(self.data, start_dim, end_dim)
  
  def F(self) -> list:
    return _flatten(self.data)
  
  def numel(self) -> int:
    out = 1
    for dim in self.shape:
      out = out * dim
    return out
  
  def size(self) -> tuple:
    return tuple(get_shape(self.data))
  
  def T(self):
    return array(list(map(list, zip(*self.data))), dtype=self.dtype)
  
  def transpose(self, dim0:int, dim1:int):
    def _re_transpose(data, dim0, dim1, ndim, depth=0):
      if depth == ndim - 2:
        return [list(row) for row in zip(*data)]
      else:
        return [_re_transpose(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

    if dim0 >= self.ndim or dim1 >= self.ndim:
      raise ValueError("Transpose dimensions out of range")
    return array(_re_transpose(self.data, dim0, dim1, self.ndim), dtype=self.dtype)

  def reshape(self, new_shape:tuple) -> List["array"]:
    new_shape = new_shape if isinstance(new_shape, tuple) else tuple(new_shape,)
    return array(reshape(self.data, new_shape), dtype=self.dtype)

  def __add__(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other)
    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b

    target_shape, requires_broadcasting = broadcasted_shape(self.shape, other.shape)
    req_conversion = handle_float(self, other)
    if req_conversion:
      self.dtype = array.float32
      other.dtype = array.float32
    
    if requires_broadcasting:
      self = array(broadcasted_array(self.data, target_shape), dtype=self.dtype)
      other = array(broadcasted_array(other.data, target_shape), dtype=other.dtype)
    
    if self.shape == other.shape:
      return array(_add(self.data, other.data), dtype=self.dtype)
    else:
      raise ValueError("shapes are incompatible for operation")

  def __mul__(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other)
    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    
    target_shape, requires_broadcasting = broadcasted_shape(self.shape, other.shape)
    req_conversion = handle_float(self, other)
    if req_conversion:
      self.dtype = array.float32
      other.dtype = array.float32

    if requires_broadcasting:
      self = array(broadcasted_array(self.data, target_shape), dtype=self.dtype)
      other = array(broadcasted_array(other.data, target_shape), dtype=other.dtype)

    if self.shape == other.shape:
      return array(_mul(self.data, other.data), dtype=self.dtype)
    else:
      raise ValueError("shapes are incompatible for operation")
  
  def __matmul__(self, other:List["array"]) -> List["array"]:
    other = other if isinstance(other, array) else array(other, dtype=self.dtype)
    if self.shape[-1] != other.shape[-2]:
      raise ValueError("Matrices have incompatible dimensions for matmul")

    req_conversion = handle_float(self, other)
    if req_conversion:
      self.dtype = array.float32
      other.dtype = array.float32

    def _remul(a, b):
      if len(a.shape) == 2 and len(b.shape) == 2:
        out = _zeros((len(a.data), len(b.data[0])))
        b_t = b.T().data
        for i in range(len(a.data)):
          for j in range(len(b_t)):
            out[i][j] = sum(a.data[i][k] * b_t[j][k] for k in range(len(a.data[0])))
        return out
      else:
        out_shape = a.shape[:-1] + (b.shape[-1],)
        out = _zeros(out_shape)
        for i in range(len(a.data)):
          out[i] = _remul(array(a.data[i]), array(b.data[i]))
        return out

    return array(_remul(self, other), dtype=array.float32)

  def __sub__(self, other:List["array"]) -> List["array"]:
    return self + (-other)

  def __rsub__(self, other:List["array"]) -> List["array"]:
    return other + (-self)
  
  def __rmul__(self, other:List["array"]) -> List["array"]:
    return other * self
  
  def __truediv__(self, other:List["array"]) -> List["array"]:
    return self * other ** -1
  
  def rtruediv(self, other:List["array"]) -> List["array"]:
    return other * self ** -1

  def __neg__(self) -> List["array"]:
    def _neg(a):
      if isinstance(a, list):
        return [_neg(_a) for _a in a]
      return -a
    return array(_neg(self.data), dtype=self.dtype)

  def __pow__(self, pow:Union[int, float], eps:float=1e6) -> List["array"]:
    assert isinstance(pow, (int, float)), "power exponent is of incompatible datatype"

    def _pow(data, pow):
      if isinstance(data, list):
        return [_pow(d, pow) for d in data]
      if data == 0:
        data = eps
      return math.pow(data, pow)

    return array(_pow(self.data, pow), dtype=array.float32)

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
  
  def relu(self) -> List["array"]:
    def _apply(data):
      if isinstance(data, list):
        return [_apply(sub_data) for sub_data in data]
      else:
        return relu(data)
    return array(_apply(self.data), dtype=array.float32)
  
  def tanh(self) -> List["array"]:
    def _apply(data):
      if isinstance(data, list):
        return [_apply(sub_data) for sub_data in data]
      else:
        return tanh(data)
    return array(_apply(self.data), dtype=array.float32)
  
  def sigmoid(self) -> List["array"]:
    def _apply(data):
      if isinstance(data, list):
        return [_apply(sub_data) for sub_data in data]
      else:
        return sigmoid(data)
    return array(_apply(self.data), dtype=array.float32)
  
  def gelu(self) -> List["array"]:
    def _apply(data):
      if isinstance(data, list):
        return [_apply(sub_data) for sub_data in data]
      else:
        return gelu(data)
    return array(_apply(self.data), dtype=array.float32)

  def mean(self, axis:Optional[int]=None, keepdims:bool=False) -> list[float]:
    if axis is None:
      flat_array = self.F()
      mean_val = sum(flat_array) / len(flat_array)
      if keepdims:
        return [[mean_val]]
      return mean_val
    else:
      return mean_axis(self.data, axis, keepdims)

  def var(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> list[float]:
    if axis is None:
      flat_array = _flatten(self.data)
      mean_value = sum(flat_array) / len(flat_array)
      variance = sum((x - mean_value) ** 2 for x in flat_array) / (len(flat_array) - ddof)
      if keepdims:
        return [[variance]]
      return variance
    else:
      mean_values = self.mean(axis=axis)
      return var_axis(self.data, mean_values, axis, ddof, keepdims)

  def std(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> list[float]:
    variance = self.var(axis=axis, ddof=ddof, keepdims=keepdims)
    def _std(var):
      if isinstance(var, list):
        return [_std(sub) for sub in var]
      return math.sqrt(var)
    if keepdims:
      return [[math.sqrt(x)] for x in _flatten(variance)]
    else:
      return _std(variance)
  
  def unsqueeze(self, dim:int=0):
    return array(_unsqueeze(self.data, dim), dtype=self.dtype)
  
  def squeeze(self, dim:int=0):
    if dim is not None and (dim<0 or dim>=self.ndim):
      raise IndexError(f"Dimension out of range (expected to be in range of {self.ndim} dimensions)")

    return array(_squeeze(self.data, dim), dtype=self.dtype)