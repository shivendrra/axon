import unittest
import numpy as np
import axon
from axon import array

class TestArray(unittest.TestCase):
  
  def setUp(self):
    self.data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
    ]
    self.custom_array = array(self.data, dtype=axon.float32)
    self.numpy_array = np.array(self.data, dtype=np.float32)
  
  def test_addition(self):
    np_result = self.numpy_array + 1
    custom_result = self.custom_array + array([[1]*3]*3)
    np.testing.assert_array_equal(np_result, custom_result.tolist())
  
  def test_multiplication(self):
    np_result = self.numpy_array * 2
    custom_result = self.custom_array * array([[2]*3]*3)
    np.testing.assert_array_equal(np_result, custom_result.tolist())
  
  def test_transpose(self):
    np_result = self.numpy_array.T
    custom_result = self.custom_array.T
    np.testing.assert_array_equal(np_result, custom_result.tolist())
  
  def test_mean(self):
    np_result = self.numpy_array.mean()
    custom_result = self.custom_array.mean()
    self.assertAlmostEqual(np_result, custom_result, places=5)

  def test_var(self):
    np_result = self.numpy_array.var()
    custom_result = self.custom_array.var()
    self.assertAlmostEqual(np_result, custom_result, places=5)

  def test_sum(self):
    np_result = self.numpy_array.sum()
    custom_result = self.custom_array.sum().data[0]
    self.assertEqual(np_result, custom_result)

  def test_log(self):
    np_result = np.log(self.numpy_array + 1)
    custom_result = self.custom_array + 1
    custom_log_result = custom_result.log()
    np.testing.assert_array_almost_equal(np_result, custom_log_result.tolist(), decimal=5)

  def test_clip(self):
    np_result = np.clip(self.numpy_array, 3, 7)
    custom_result = self.custom_array.clip(3, 7)
    np.testing.assert_array_equal(np_result, custom_result.tolist())

if __name__ == '__main__':
  unittest.main()