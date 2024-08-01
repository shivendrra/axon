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
# print(f)

from axon.micro import Tensor

a, b = [[1, 5, -6], [1, 6, -3]], [[-2, 0, -6], [7, -2, 0]]
a, b = Tensor(a), Tensor(b)

c = a + b
d = c.relu()
e = c.tanh()
f = c.swiglu()

print(c)
print(d)
print(e)
print(f)