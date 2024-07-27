#include "Value.h"

Value::Value(double data) : data(data), grad(0.0), exp(0.0) {
  _backward = noop_backward;
}

void Value::noop_backward(Value* v) {}

std::shared_ptr<Value> Value::add(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto out = std::make_shared<Value>(a->data + b->data);
  out->_prev = {a.get(), b.get()};
  out->_backward = add_backward;
  return out;
}

void Value::add_backward(Value* v) {
  v->_prev[0]->grad += v->grad;
  v->_prev[1]->grad += v->grad;
}

std::shared_ptr<Value> Value::mul(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto out = std::make_shared<Value>(a->data * b->data);
  out->_prev = {a.get(), b.get()};
  out->_backward = mul_backward;
  return out;
}

void Value::mul_backward(Value* v) {
  v->_prev[0]->grad += v->_prev[1]->data * v->grad;
  v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

std::shared_ptr<Value> Value::pow_val(const std::shared_ptr<Value>& a, double exp) {
  auto out = std::make_shared<Value>(std::pow(a->data, exp));
  out->_prev = {a.get()};
  out->exp = exp;
  out->_backward = pow_backward;
  return out;
}

void Value::pow_backward(Value* v) {
  v->_prev[0]->grad += v->exp * std::pow(v->_prev[0]->data, v->exp - 1) * v->grad;
}

std::shared_ptr<Value> Value::negate(const std::shared_ptr<Value>& a) {
  return mul(a, std::make_shared<Value>(-1.0));
}

std::shared_ptr<Value> Value::sub(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return add(a, negate(b));
}

std::shared_ptr<Value> Value::truediv(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return mul(a, pow_val(b, -1.0));
}

std::shared_ptr<Value> Value::relu(const std::shared_ptr<Value>& a) {
  auto out = std::make_shared<Value>(a->data > 0 ? a->data : 0);
  out->_prev = {a.get()};
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