import axon
from axon import array

X = array(axon.random.randn(2, 100), dtype=axon.float64)
print(X)

X = array(axon.random.randint(low=2, high=100, size=(4, 3)), dtype=axon.float16)
print(X)

axon.random.seed(400)
X = array(axon.random.rand(2, 100), dtype=axon.float16)
print(X)