from typing import *
from .utils import _zeros

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

def broadcast_shape(shape1:tuple, shape2:tuple) -> tuple:
  res_shape = []
  if shape1 == shape2:
    return shape1, False
  
  max_len = max(len(shape1), len(shape2))
  shape1 = [1] * (max_len - len(shape1)) + shape1
  shape2 = [1] * (max_len - len(shape2)) + shape2

  for dim1, dim2 in zip(shape1, shape2):
    if dim1 != dim2 and dim1 != 1 and dim2 != 1:
      raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
    res_shape.append(max(dim1, dim2))
  return tuple(res_shape), True

def broadcast(array, target_shape):
  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array

  def expand_dims(array, current_shape, target_shape):
    if not current_shape:
      return array

    if len(current_shape) < len(target_shape):
      array = [array]
      current_shape = [1] + current_shape
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
  assert type(new_shape) == tuple, "new shape must be a tuple"
  def _shape_numel(shape):
    numel = 1
    for ele in shape:
      numel *= ele
    return numel
  
  if _shape_numel(new_shape) != _shape_numel(get_shape(data)):
    raise ValueError(f"Shapes {new_shape} & {get_shape(data)} incompatible for reshaping")
  else:
    def _reshape(data, new_shape):
      flatten_data = flatten(data)
      target = _zeros(shape=new_shape)
      idx = [0]

      def __populate(target, shape):
        if len(shape) == 1:
          for i in range(shape[0]):
            target[i] = flatten_data[idx[0]]
            idx[0] += 1
        else:
          for i in range(shape[0]):
            __populate(target[i], shape[1:])

      __populate(target, list(new_shape))
      return target
  return _reshape(data, new_shape)

def unsqueeze(data:list, dim:int=0) -> list:
  if dim == 0:
    return [item for sublist in data for item in unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [unsqueeze(d, dim-1) for d in data]
    return [data]

def squeeze(data:list, dim:Union[int, None]) -> list:
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

def swap_axes(array: list, axis1: int, axis2: int) -> list:
    def recursive_swap(sub_array, current_axis):
        if current_axis == min(axis1, axis2):
            sub_array = [list(x) for x in zip(*sub_array)]
        if current_axis == max(axis1, axis2) - 1:
            return sub_array
        return [recursive_swap(sub, current_axis + 1) for sub in sub_array]

    ndim = len(get_shape(array))

    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim

    if axis1 >= ndim or axis2 >= ndim:
        raise ValueError("Axis out of bounds")

    if axis1 == axis2:
        return array

    return recursive_swap(array, 0)