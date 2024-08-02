# from axon import value

# a = value(2)
# b = value(3)

# c = a + b
# d = a * b
# e = c.relu()
# f = d ** 2.0

# f.backward()

# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print()

def get_shape(arr):
  if isinstance(arr, list):
    return [len(arr)] + get_shape(arr[0])
  else:
    return []

def flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  else:
    return [data]

def re_flat(input_tensor, start_dim=0, end_dim=-1):
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return flatten(data)
    else:
      return data
  if end_dim == -1:
    end_dim = len(get_shape(input_tensor)) - 1
  return _recurse_flatten(input_tensor, 0)

def transpose(data):
  if not data:
    return []
  if isinstance(data[0], list):
    return list(map(list, zip(*data)))
  return [[x] for x in data]

def sum_along_axis(data, axis):
  def recursive_sum(data, current_dim):
    if current_dim == axis:
      if isinstance(data[0], list):
        return [sum(sublist) if isinstance(sublist, list) else sublist for sublist in data]
      else:
        return sum(data)
    if isinstance(data[0], list):
      return [recursive_sum(sublist, current_dim + 1) for sublist in data]
    else:
      return data

  if axis == 0:
    transposed_data = transpose(data)
    return [sum_along_axis(sublist, axis) for sublist in transposed_data]

  return recursive_sum(data, 0)

# Example usage:
a = [[[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
     [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]],
     [[[1, -2, 9], [4, -2, 1]], [[0, -2, 9], [1, 3, 5]]]]

print("Original array:", a)
print(sum_along_axis(a))
print("Sum of array along axis 0:", sum_along_axis(a, axis=0))
print("Sum of array along axis 1:", sum_along_axis(a, axis=1))
print("Sum of array along axis 2:", sum_along_axis(a, axis=2))
print("Sum of array along axis 3:", sum_along_axis(a, axis=3))

# print("Original array:", a)
# print("Flattened array:", flatten(a))
# print("Shape of array:", get_shape(a))
# print("Re-flattened array (1, -1):", re_flat(a, 1, -1))
# print("Shape of re-flattened array (2, -1):", get_shape(re_flat(a, 2, -1)))
# print("Sum of array:", sum_axis(a))
# print("Sum of array along axis 0:", sum_axis(a, axis=0))
# print("Sum of array along axis 1:", sum_axis(a, axis=1))
# print("Sum of array along axis 2:", sum_axis(a, axis=2))
# print("Sum of array along axis 3:", sum_axis(a, axis=3))

# import numpy as np
# a = np.array(a)
# print(a.sum())
# print(a.sum(axis=0))
# print(a.sum(axis=1))
# print(a.sum(axis=1).shape)
# print(a.sum(axis=2))
# print(a.sum(axis=3))