from typing import *
from .utils.shape import shape

class array:
  def __init__(self, *data, dtype=None) -> None:
    self.data = data
    self.shape = shape()