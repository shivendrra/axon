def get_shape(arr):
  if isinstance(arr, list):
    return [len(arr),] + get_shape(arr[0])
  else:
    return []

def _flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in _flatten(sublist)]
  else:
    return [data]

def __flatten(input_tensor, start_dim=0, end_dim=-1):
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return __flatten(data)
    else:
      return data
  
  if end_dim == -1:
    end_dim = len(input_tensor) - 1

  return [_recurse_flatten(input_tensor, 0)]