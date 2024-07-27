#include "engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>

Value::Value(double data) : data(data), grad(0.0), exp(0.0) {
  _backward = noop_backward;
}

void Value::noop_backward(Value* v) {}

Value* Value::add(Value* a, Value* b) {
  Value* out = new Value(a->data + b->data);
  out->_prev = {a, b};
  out->_backward = add_backward;
  return out;
}

void Value::add_backward(Value* v) {
  v->_prev[0]->grad += v->grad;
  v->_prev[1]->grad += v->grad;
}

Value* Value::mul(Value* a, Value* b) {
  Value* out = new Value(a->data * b->data);
  out->_prev = {a, b};
  out->_backward = mul_backward;
  return out;
}

void Value::mul_backward(Value* v) {
  v->_prev[0]->grad += v->_prev[1]->data * v->grad;
  v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

Value* Value::pow_val(Value* a, double exp) {
  Value* out = new Value(std::pow(a->data, exp));
  out->_prev = {a};
  out->exp = exp;
  out->_backward = pow_backward;
  return out;
}

void Value::pow_backward(Value* v) {
  v->_prev[0]->grad += v->exp * std::pow(v->_prev[0]->data, v->exp - 1) * v->grad;
}

Value* Value::negate(Value* a) {
  return mul(a, new Value(-1.0));
}

Value* Value::sub(Value* a, Value* b) {
  return add(a, negate(b));
}

Value* Value::truediv(Value* a, Value* b) {
  return mul(a, pow_val(b, -1.0));
}

Value* Value::relu(Value* a) {
  Value* out = new Value(a->data > 0 ? a->data : 0);
  out->_prev = {a};
  out->_backward = relu_backward;
  return out;
}

void Value::relu_backward(Value* v) {
  v->_prev[0]->grad += (v->data > 0) * v->grad;
}

void Value::build_topo(Value* v, std::vector<Value*>& topo, std::vector<Value*>& visited) {
  if (std::find(visited.begin(), visited.end(), v) == visited.end()) {
    visited.push_back(v);
    for (auto child : v->_prev) {
      build_topo(child, topo, visited);
    }
    topo.push_back(v);
  }
}

void Value::backward(Value* v) {
  std::vector<Value*> topo;
  std::vector<Value*> visited;
  build_topo(v, topo, visited);

  v->grad = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward(*it);
  }
}

void Value::print_value() const {
  std::cout << "Value(data=" << data << ", grad=" << grad << ")\n";
}