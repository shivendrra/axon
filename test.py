from axon import value

a = value(2)
b = value(3)

c = a + b
d = a * b
e = c.relu()
f = d ** 2.0

f.backward()

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)