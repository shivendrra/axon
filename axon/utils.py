import random
from .helpers.dtype import *
from typing import *
from .base import array

def zeros(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if len(shape) == 1:
    return [0] * shape[0]
  return array([zeros(shape[1:]) for _ in range(shape[0])], dtype=dtype)

def ones(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if len(shape) == 1:
    return [1] * shape[0]
  return array([ones(shape[1:]) for _ in range(shape[0])], dtype=dtype)

def randint(low:int, high:int, size:int=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if size is None:
    return random.randint(low, high)
  else:
    return array([random.randint(low, high) for _ in range(size)], dtype=dtype)

def arange(start:int=0, end:int=10, step:int=1) -> list:
  return array([start + i * step for i in range(int((end-start)/step))], dtype=None)

def randn(domain:tuple=(1, -1), shape:tuple=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if len(shape) == 1:
    return [random.uniform(domain[0], domain[1]) for _ in range(shape[0])]
  else:
    return array([randn(domain=domain, shape=shape[1:]) for _ in range(shape[0])], dtype=dtype)

def zeros_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if isinstance(arr, list):
    return array([zeros_like(elem) for elem in arr], dtype=dtype)
  elif isinstance(arr, tuple):
    return tuple(zeros_like(elem) for elem in arr)
  else:
    return 0

def ones_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  if isinstance(arr, list):
    return array([ones_like(elem) for elem in arr], dtype=dtype)
  elif isinstance(arr, tuple):
    return tuple(ones_like(elem) for elem in arr)
  else:
    return 1