from .base import array
from .helpers.utils import _zeros
from .helpers.shape import _squeeze, _unsqueeze, get_shape, _get_element
from typing import *

def matmul(a:Union[array, list], b:Union[array, list]) -> array:
  a = a if isinstance(a, array) else array(a)
  b = b if isinstance(b, array) else array(b)
  return (a @ b).data

def dot(a:Union[array, list], b:Union[array, list]) -> array:
  a = a if isinstance(a, array) else array(a)
  b = b if isinstance(b, array) else array(b)
  return a.dot(b)

def stack(data: tuple[array, array], axis: int=0) -> array:
  if not data:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = data[0].shape
  for d in data:
    if d.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  
  # new shape after stacking & initilization
  new_shape = list(base_shape[:])
  new_shape.insert(axis, len(data))
  new_data = _zeros(new_shape)

  def insert_data(new_data, arrays, axis, indices=[]):
    if len(indices) == len(new_shape):
      for idx, array in enumerate(arrays):
        data_idx = indices[:]
        data_idx[axis] = idx
        sub_arr = new_data
        for k in data_idx[:-1]:
          sub_arr = sub_arr[k]
        sub_arr[data_idx[-1]] = _get_element(array.data, indices[:axis] + indices[axis+1:])
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, arrays, axis, indices + [i])
  
  insert_data(new_data, data, axis)
  return array(new_data, dtype=data[0].dtype)

def concat(data: tuple[array, array], axis: int=0) -> array:
  if not data:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = list(data[0].shape) # shape of first array for target array
  for arr in data:
    if list(arr.shape)[:axis] + list(arr.shape)[axis+1:] != base_shape[:axis] + base_shape[axis+1:]:
      raise ValueError("All input arrays must have the same shape except for the concatenation axis")
  
  new_shape = base_shape[:]
  new_shape[axis] *= len(data)
  new_data = _zeros(new_shape)

  def set_element(data, indices, value):
    for idx in indices[:-1]:
      data = data[idx]
    data[indices[-1]] = value

  def insert_data(new_data, arrays, axis, indices=[]):
    if len(indices) == len(new_shape):
      current_offset = 0
      for array in arrays:
        if current_offset <= indices[axis] < current_offset + array.shape[axis]:
          local_indices = indices[:]
          local_indices[axis] -= current_offset
          ele = _get_element(array.data, local_indices)
          set_element(new_data, indices, ele)
          break
        current_offset += array.shape[axis]
      return
      
    for i in range(new_shape[len(indices)]):
      insert_data(new_data, arrays, axis, indices + [i])
  
  insert_data(new_data, data, axis)
  return array(new_data, dtype=data[0].dtype)

def split(data:Union[array, list], idx:int, axis:Optional[int]=None) -> list:
  def _get_slices(start_idx, end_idx, data):
    slices = []
    for start, end in zip(start_idx, end_idx):
      slices.append(data[start:end])
    return slices
  
  if isinstance(idx, int):
    N = idx
    total_len = len(data) if axis == 0 else len(data[0])
    if total_len % N != 0:
      raise ValueError("Array split doesn't results in an equal division")
    step = total_len // N
    indices = [i*step for i in range(1, N)]
  else:
    indices = idx
  
  start_idx = [0] + indices
  end_idx = indices + [len(data) if axis==0 else len(data)]

  if axis == 0:
    return _get_slices(start_idx, end_idx, data)
  else:
    result = []
    for row in data:
      result.append(_get_slices(start_idx, end_idx, row))
    return [list(col) for col in zip(*result)]
  
def mean(data, axis:Optional[int]=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, array) else array(data, dtype)
  return data.mean(axis=axis, keepdims=keepdims)

def var(data, axis:Optional[int]=None, ddof:int=0, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, array) else array(data, dtype)
  return data.var(axis=axis, ddof=ddof, keepdims=keepdims)

def std(data, axis:Optional[int]=None, ddof:int=0, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None, keepdims:bool=False) -> Union[list, float, int]:
  data = data if isinstance(data, array) else array(data, dtype)
  return data.std(axis=axis, ddof=ddof, keepdims=keepdims)

def squeeze(*data, dim:int=0):
  for _data in data:
    if dim is not None and (dim<0 or dim>=len(get_shape(_data))):
      raise IndexError(f"Dimension out of range (expected to be in range of {len(get_shape(_data))} dimensions)")
    else:
      return _squeeze(_data, dim)

def unsqueeze(*data, dim:int=0):
  for _data in data:
    return _unsqueeze(_data, dim)

def clip(data, min, max, out=None):
  data = data if isinstance(data, array) else array(data)
  if out is not None:
    return data.clip(min_value=min, max_value=max)
  else:
    out = data.clip(min_value=min, max_value=max)
    return out