def shape(arr):
  if isinstance(arr, list):
    return [len(arr),] + shape(arr[0])
  else:
    return []