from .shapes import transpose, get_shape
from .utils import _zeros

def sum_axis0(data):
  if not isinstance(data[0], list):
    return sum(data)

  result = []
  for i in range(len(data[0])):
    result.append(sum_axis0([d[i] for d in data]))
  return result

def mean_axis0(data):
  if not isinstance(data[0], list):
    return sum(data) / len(data)

  result = []
  for i in range(len(data[0])):
    result.append(mean_axis0([d[i] for d in data]))
  return result

def var_axis0(data, ddof=0):
  mean_values = mean_axis0(data)
  if not isinstance(data[0], list):
    return sum((x - mean_values) ** 2 for x in data) / (len(data) - ddof)

  result = []
  for i in range(len(data[0])):
    result.append(var_axis0([d[i] for d in data], ddof))
  return result

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

def sum_axis(data, axis, keepdims):
  if axis==0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in transposed]
  else:
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in data]
  if keepdims:
    mean_vals = [mean_vals]
  return mean_vals

def matmul(a, b):
  def _remul(a, b):
    if len(get_shape(a)) == 2 and len(get_shape(b)) == 2:
      out = _zeros((len(a), len(b[0])))
      b_t = transpose(b)
      for i in range(len(a)):
        for j in range(len(b_t)):
          out[i][j] = sum(a[i][k] * b_t[j][k] for k in range(len(a[0])))
      return out
    else:
      out_shape = get_shape(a)[:-1] + (get_shape(b)[-1],)
      out = _zeros(out_shape)
      for i in range(len(a)):
        out[i] = _remul((a[i]), (b[i]))
      return out

  if get_shape(a)[-1] != get_shape(b)[-2]:
    raise ValueError("Matrices have incompatible dimensions for matmul")
  return _remul(a, b)