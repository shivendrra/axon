from typing import List
from ..helpers.shape import reshape

class ContiguousOps:
  def __init__(self, array_instance):
    # linked the tensor instance to use its shape, size, and data
    self.array_instance = array_instance

  def is_contiguous(self) -> bool:
    """check if the tensor's memory layout is contiguous."""
    expected_stride = 1
    # need to check the strides in reverse order
    for dim, stride in zip(reversed(self.array_instance.shape), reversed(self.array_instance.stride)):
      if stride != expected_stride:
        return False
      expected_stride *= dim
    return True

  def make_contiguous(self) -> None:
    """ensure the tensor is contiguous by rearranging its data if necessary."""
    if not self.is_contiguous():
      # ff the tensor is not contiguous, we need to reorder the data into a contiguous format
      contiguous_data = self._reshape(self.array_instance.data, self.array_instance.shape)
      self.array_instance.data = contiguous_data
      # update the stride to reflect the contiguous layout
      self.array_instance.stride = self.compute_stride(self.array_instance.shape)

  def compute_stride(self, shape: List[int]) -> List[int]:
    """compute the strides for a given shape to ensure a contiguous memory layout."""
    strides = [1]
    for size in reversed(shape[:-1]):
      strides.append(strides[-1] * size)
    return list(reversed(strides))

  def _reshape(self, data, shape: List[int]) -> List:
    # reshape the data
    return reshape(data, shape)