def get_shape(arr):
  if isinstance(arr, list):
    return [len(arr),] + get_shape(arr[0])
  else:
    return []

def _get_element(data, indices):
  for idx in indices:
    data = data[idx]
  return data

def _flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in _flatten(sublist)]
  else:
    return [data]

def re_flat(input_tensor, start_dim=0, end_dim=-1):
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return _flatten(data)
    else:
      return data
  
  if end_dim == -1:
    end_dim = len(input_tensor) - 1

  return _recurse_flatten(input_tensor, 0)

def mean_axis(data, axis, keepdims):
  if axis == 0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    mean_vals = [mean_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) / len(d) for d in transposed]
  else:
    mean_vals = [mean_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) / len(d) for d in data]
  if keepdims:
    mean_vals = [mean_vals]
  return mean_vals

def var_axis(data, mean_values, axis, ddof, keepdims):
  if axis == 0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    variance = [var_axis(d, mean_values[i], axis - 1, ddof, keepdims) if isinstance(d[0], list) else sum((x - mean_values[i]) ** 2 for x in d) / (len(d) - ddof) for i, d in enumerate(transposed)]
  else:
    variance = [var_axis(d, mean_values[i], axis - 1, ddof, keepdims) if isinstance(d[0], list) else sum((x - mean_values[i]) ** 2 for x in d) / (len(d) - ddof) for i, d in enumerate(data)]
  if keepdims:
    variance = [variance]
  return variance

def broadcasted_shape(shape1, shape2):
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
  return res_shape, True

def _unsqueeze(data, dim=0):
  if dim == 0:
    return [item for sublist in data for item in _unsqueeze(sublist)] if isinstance(data, list) else [data]
  else:
    if isinstance(data, list):
      return [_unsqueeze(d, dim-1) for d in data]
    return [data]

def _squeeze(data, dim):
  if dim is None:
    if isinstance(data, list):
      squeezed = [_squeeze(d, None) for d in data]
      return squeezed if len(squeezed) > 1 else squeezed[0]
    return data
  if isinstance(data, list):
    if dim == 0:
      return data[0] if len(data) == 1 else data
    return [_squeeze(d, dim - 1) for d in data]
  return data

def broadcasted_array(array, target_shape):
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

def reshape(data, new_shape):
  flat_data = _flatten(data)
  shape_size = _shape_size(new_shape)
  if shape_size != len(flat_data):
    raise ValueError("Total size of new array must be unchanged")

  def _recursive_reshape(data, shape):
    if len(shape) == 1:
      return data[:shape[0]]
    size = shape[0]
    sub_size = _shape_size(shape[1:])
    return [_recursive_reshape(data[i * sub_size:(i + 1) * sub_size], shape[1:]) for i in range(size)]

  return _recursive_reshape(flat_data, new_shape)

def _shape_size(shape):
  size = 1
  for dim in shape:
    size *= dim
  return size