from axon import Value

a = Value(2)
b = Value(3)

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