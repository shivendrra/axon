def get_shape(data):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  return []

def flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  return [data]

def re_flatten(data, start_dim=0, end_dim=-1):
  def _re_flat(data, current_dim):
    if current_dim < start_dim:
      return [_re_flat(item, current_dim+1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return flatten(data)
    return data
  
  if end_dim == -1:
    end_dim = len(data) - 1
  return re_flatten(data)

def broadcast_shape(shape1, shape2):
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

def unsqueeze(data, dim=0):
  if dim == 0:
    if isinstance(data, list):
      return [item for sublist in data for item in unsqueeze(sublist)]
    else:
      [data]
  else:
    if isinstance(data, list):
      return [unsqueeze(d, dim-1) for d in data]
    else:
      return [data]

