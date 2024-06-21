def shape(arr):
  if isinstance(arr, list):
    return [len(arr),] + shape(arr[0])
  else:
    return []

a = [[1, 4, 5], [1, 4, 5]]
print(shape(a))