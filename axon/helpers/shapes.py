from typing import *

def get_shape(data:list) -> list:
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def flatten(data:list) -> list:
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  else:
    return [data]

def flatten_recursive(data:list, start_dim:int=0, end_dim:int=-1) -> list:
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return flatten(data)
    else:
      return data
  if end_dim == -1:
    end_dim = len(get_shape(data)) - 1
  return _recurse_flatten(data, 0)

def transpose(data:list) -> list:
  return list(map(list, zip(*data)))

def transpose_recursive(data:list, dim:int) -> list:
  raise NotImplementedError("not written")

def swap_axes(data:list, dim0:int, dim1:int, ndim:int, depth:int=0) -> list:
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [swap_axes(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

def broadcast_shape(shape1:tuple, shape2:tuple) -> tuple:
  res_shape = []
  if shape1 == shape2:
    return shape1, False
  
  else:
    max_len = max(len(shape1), len(shape2))
    shape1 = [1] * (max_len - len(shape1)) + shape1
    shape2 = [1] * (max_len - len(shape2)) + shape2

    for dim1, dim2 in zip(shape1, shape2):
      if dim1 != dim2 and dim2 != 1 and dim1 != 1:
        raise ValueError(f"shape {shape1} and {shape2} are not compatible for broadcasting")
      res_shape.append(dim1, dim2)
    return res_shape, True

def broadcast(array, target_shape):
  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array

  def expand_dims(array, current_shape, target_shape):
    if len(current_shape) < len(target_shape):
      array = [array]
      current_shape = [1,] + current_shape
    if current_shape == target_shape:
      return array

    if current_shape[0] == 1:
      array = array * target_shape[0]
    result = []
    for subarray in array:
      result.append(expand_dims(subarray, current_shape[1:], target_shape[1:]))
    return result

  return expand_dims(array, current_shape, target_shape)

def reshape(data:list, new_shape:tuple) -> list:
  raise NotImplementedError("Not yet written")

def unsqueeze(data, dim=0):
  if dim == 0:
    return [item for sublist in data for item in unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [unsqueeze(d, dim-1) for d in data]
    return [data]

def squeeze(data, dim):
  if dim is None:
    if isinstance(data, list):
      squeezed = [squeeze(d, None) for d in data]
      return squeezed if len(squeezed) > 1 else squeezed[0]
    return data
  if isinstance(data, list):
    if dim == 0:
      return data[0] if len(data) == 1 else data
    return [squeeze(d, dim - 1) for d in data]
  return data