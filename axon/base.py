import engine

class value:
  def __init__(self, data):
    self.value = engine.Value(data)

  def add(self, other):
    return value(engine.Value.add(self.value, other.value))

  def mul(self, other):
    return value(engine.Value.mul(self.value, other.value))

  def pow_val(self, exp):
    return value(engine.Value.pow_val(self.value, exp))

  def negate(self):
    return value(engine.Value.negate(self.value))

  def sub(self, other):
    return value(engine.Value.sub(self.value, other.value))

  def truediv(self, other):
    return value(engine.Value.truediv(self.value, other.value))

  def relu(self):
    return value(engine.Value.relu(self.value))

  def backward(self):
    engine.Value.backward(self.value)

  def __repr__(self):
    self.value.print_value()