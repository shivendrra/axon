#include "engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#define PI 3.14159265358979323846

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

Value* Value::tanh(Value* a) {
  Value* out = new Value((std::exp(a->data) - std::exp(-a->data)) / (std::exp(a->data) + std::exp(-a->data)));
  out->_prev = {a};
  out->_backward = tanh_backward;
  return out;
}

void Value::tanh_backward(Value* v) {
  v->_prev[0]->grad += (1 - (v->data * v->data)) * v->grad;
}

Value* Value::sigmoid(Value* a) {
  Value* out = new Value(1 / (1 + std::exp(-a->data)));
  out->_prev = {a};
  out->_backward = sigmoid_derivative;
  return out;
}

void Value::sigmoid_derivative(Value* v) {
  v->_prev[0]->grad += (v->data * (1 - v->data)) * v->grad;
}

Value* Value::gelu(Value* a) {
  double phi = 0.5 * (1 + std::erf(a->data / std::sqrt(2)));
  Value* out = new Value(a->data * phi);
  out->_prev = {a};
  out->_backward = gelu_derivative;
  return out;
}

void Value::gelu_derivative(Value* v) {
  Value* a = v->_prev[0];
  double phi = 0.5 * (1 + std::erf(a->data / std::sqrt(2)));
  double pdf = std::exp(-0.5 * a->data * a->data) / std::sqrt(2 * PI);
  a->grad += (phi + a->data * pdf) * v->grad;
}

Value* Value::silu(Value* a) {
  double sigmoid_a = 1 / (1 + std::exp(-a->data));
  Value* out = new Value(a->data * sigmoid_a);
  out->_prev = {a};
  out->_backward = silu_derivative;
  return out;
}

void Value::silu_derivative(Value* v) {
  Value* a = v->_prev[0];
  double sigmoid_a = 1 / (1 + std::exp(-a->data));
  a->grad += (sigmoid_a * (1 + a->data * (1 - sigmoid_a))) * v->grad;
}

Value* Value::swiglu(Value* a) {
  double sigmoid_a = 1 / (1 + std::exp(-a->data));
  double glu_a = sigmoid_a * a->data;
  Value* out = new Value(a->data * sigmoid_a * glu_a);
  out->_prev = {a};
  out->_backward = swiglu_derivative;
  return out;
}

void Value::swiglu_derivative(Value* v) {
  Value* a = v->_prev[0];
  double sigmoid_a = 1 / (1 + std::exp(-a->data));
  double glu_a = sigmoid_a * a->data;
  a->grad += (sigmoid_a * (1 + a->data * (1 - sigmoid_a)) * glu_a + a->data * sigmoid_a * (1 - sigmoid_a) * a->data) * v->grad;
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

std::string Value::repr() const {
  return "Value(data=" + std::to_string(data) + ", grad=" + std::to_string(grad) + ")";
}

double Value::get_data() const {
  return data;
}

double Value::get_grad() const {
  return grad;
}