# def get_shape(arr):
#   if isinstance(arr, list):
#     return [len(arr)] + get_shape(arr[0])
#   else:
#     return []

def flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  else:
    return [data]

# def re_flat(input_tensor, start_dim=0, end_dim=-1):
#   def _recurse_flatten(data, current_dim):
#     if current_dim < start_dim:
#       return [_recurse_flatten(item, current_dim + 1) for item in data]
#     elif start_dim <= current_dim <= end_dim:
#       return flatten(data)
#     else:
#       return data
#   if end_dim == -1:
#     end_dim = len(get_shape(input_tensor)) - 1
#   return _recurse_flatten(input_tensor, 0)

# def transpose(data):
#   return [list(row) for row in zip(*data)]

# def swap_axes(data:list, dim0:int, dim1:int, depth:int=0) -> list:
#   ndim = len(get_shape(data))
#   if depth == ndim - 2:
#     return [list(row) for row in zip(*data)]
#   else:
#     return [swap_axes(sub_data, dim0, dim1, depth+1) for sub_data in data]

def sum_axis(data, axis=None, keepdims=False):
  if axis==None:
    return sum(flatten(data))
  if axis == 0:
    transposed = list(map(list, zip(*data)))
    if all(isinstance(i, list) for i in transposed[0]):
      transposed = [list(map(list, zip(*d))) for d in transposed]
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in transposed]
  else:
    mean_vals = [sum_axis(d, axis - 1, keepdims) if isinstance(d[0], list) else sum(d) for d in data]
  if keepdims:
    mean_vals = [mean_vals]
  return mean_vals

a = [[[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
     [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
     [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
     [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]]]

print("mean: ", sum_axis(a))
print("mean0: ", sum_axis(a, axis=0))
print("mean1: ", sum_axis(a, axis=1))
print("mean2: ", sum_axis(a, axis=2))
print("mean3: ", sum_axis(a, axis=3))

import numpy as np

a = np.array(a)
print("mean: ", np.sum(a))
print("mean0: ", np.sum(a, axis=0))
print("mean1: ", np.sum(a, axis=1))
print("mean2: ", np.sum(a, axis=2))
print("mean3: ", np.sum(a, axis=3))