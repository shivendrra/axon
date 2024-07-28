from typing import *
import math
from copy import deepcopy

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
long = 'long'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
double = 'double'

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