from typing import *
from ..utils._random import random

def _zeros(*shape: tuple[int]) -> list:
  if not shape:
    raise ValueError("Shape must be specified.")
  if len(shape) == 1:
    return [0] * shape[0]
  return [_zeros(*shape[1:]) for _ in range(shape[0])]

def _ones(*shape: tuple[int]) -> list:
  if not shape:
    raise ValueError("Shape must be specified.")
  if len(shape) == 1:
    return [1] * shape[0]
  return [_ones(*shape[1:]) for _ in range(shape[0])]

def _randint(*shape: tuple[int], low: int, high: int) -> list:
  if low >= high:
    raise ValueError("Low must be less than high.")
  if not shape:
    return random.randint(low, high)
  if len(shape) == 1:
    return [random.randint(low, high) for _ in range(shape[0])]
  return [_randint(low, high, *shape[1:]) for _ in range(shape[0])]

def _arange(start: int = 0, end: int = 10, step: int = 1) -> list:
  if step <= 0:
    raise ValueError("Step must be greater than 0.")
  return [start + i * step for i in range(int((end - start) / step))]

def _randn(domain: Tuple[float, float] = (-1, 1), *shape: int) -> list:
  if domain[0] >= domain[1]:
    raise ValueError("Invalid domain: lower bound must be less than upper bound.")
  if not shape:
    return random.uniform(domain[0], domain[1])
  if len(shape) == 1:
    return [random.uniform(domain[0], domain[1]) for _ in range(shape[0])]
  return [_randn(domain, *shape[1:]) for _ in range(shape[0])]

def _zeros_like(arr:list) -> list:
  if isinstance(arr, list):
    return [_zeros_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(_zeros_like(elem) for elem in arr)
  else:
    return [0]

def _ones_like(arr:list) -> list:
  if isinstance(arr, list):
    return [_ones_like(elem) for elem in arr]
  elif isinstance(arr, tuple):
    return tuple(_ones_like(elem) for elem in arr)
  else:
    return [1]