#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class Value {
public:
  double data;
  double grad;
  double exp;
  std::vector<Value*> _prev;

  void (*_backward)(Value*);

  Value(double data) : data(data), grad(0.0), exp(0.0) {
    _backward = noop_backward;
  }

  static Value* add(Value* a, Value* b) {
    Value* out = new Value(a->data + b->data);
    out->_prev = {a, b};
    out->_backward = add_backward;
    return out;
  }

  static Value* mul(Value* a, Value* b) {
    Value* out = new Value(a->data * b->data);
    out->_prev = {a, b};
    out->_backward = mul_backward;
    return out;
  }

  static Value* pow_val(Value* a, double exp) {
    Value* out = new Value(std::pow(a->data, exp));
    out->_prev = {a};
    out->exp = exp;
    out->_backward = pow_backward;
    return out;
  }

  static Value* negate(Value* a) {
    return mul(a, new Value(-1.0));
  }

  static Value* sub(Value* a, Value* b) {
    return add(a, negate(b));
  }

  static Value* truediv(Value* a, Value* b) {
    return mul(a, pow_val(b, -1.0));
  }

  static Value* relu(Value* a) {
    Value* out = new Value(a->data > 0 ? a->data : 0);
    out->_prev = {a};
    out->_backward = relu_backward;
    return out;
  }

  static void build_topo(Value* v, std::vector<Value*>& topo, std::vector<Value*>& visited) {
    if (std::find(visited.begin(), visited.end(), v) == visited.end()) {
      visited.push_back(v);
      for (auto child : v->_prev) {
        build_topo(child, topo, visited);
      }
      topo.push_back(v);
    }
  }

  static void backward(Value* v) {
    std::vector<Value*> topo;
    std::vector<Value*> visited;

    build_topo(v, topo, visited);
    v->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      (*it)->_backward(*it);
    }
  }

  void print_value() const {
    std::cout << "Value(data=" << data << ", grad=" << grad << ")\n";
  }

private:
  static void noop_backward(Value* v) {}

  static void add_backward(Value* v) {
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
  }

  static void mul_backward(Value* v) {
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
  }

  static void pow_backward(Value* v) {
    v->_prev[0]->grad += v->exp * std::pow(v->_prev[0]->data, v->exp - 1) * v->grad;
  }

  static void relu_backward(Value* v) {
    v->_prev[0]->grad += (v->data > 0) * v->grad;
  }
};

int main() {
  Value* a = new Value(2.0);
  Value* b = new Value(3.0);
  Value* c = Value::add(a, b);
  Value* d = Value::mul(a, b);
  Value* e = Value::relu(c);
  Value* f = Value::pow_val(d, 2.0);

  Value::backward(f);

  a->print_value();
  b->print_value();
  c->print_value();
  d->print_value();
  e->print_value();
  f->print_value();

  delete a;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;

  return 0;
}