from .helpers.utils import _zeros, _ones, _randint, _randn, _arange, _ones_like, _zeros_like
from .dtypes.dtype import *
from typing import *
from .base import array

def zeros(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_zeros(shape), dtype=dtype if dtype is not None else 'float32')

def ones(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_ones(shape), dtype=dtype if dtype is not None else 'float32')

def randint(low:int, high:int, size:int=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_randint(low, high, size), dtype=dtype if dtype is not None else 'int32')

def arange(start:int=0, end:int=10, step:int=1) -> list:
  return array(_arange(start, end, step), dtype=None)

def randn(domain:tuple=(1, -1), shape:tuple=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_randn(domain, shape), dtype=dtype if dtype is not None else 'float32')

def zeros_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_zeros_like(arr), dtype=dtype if dtype is not None else 'float32')

def ones_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return array(_ones_like(arr), dtype=dtype if dtype is not None else 'float32')