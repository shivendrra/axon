#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Value {
  double data;
  double grad;
  struct Value** _prev;
  int _prev_size;
  void (*_backward)(struct Value*);
  double exp;
} Value;

void noop_backward(Value* v) {}

Value* new_value(double data, Value** children, int children_size) {
  Value* v = (Value*)malloc(sizeof(Value));
  v->data = data;
  v->grad = 0;
  v->_prev = children;
  v->_prev_size = children_size;
  v->_backward = noop_backward;
  v->exp = 0;
  return v;
}

void add_backward(Value*v){
  Value* a = v->_prev[0];
  Value* b = v->_prev[1];

  a->grad += v->grad;
  b->grad += v->grad;
}

Value* add(Value* a, Value* b){
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = new_value(a->data + b->data, children, 2);
  out->_backward = add_backward;
  return out;
}

void mul_backward(Value* v){
  Value* a = v->_prev[0];
  Value* b = v->_prev[1];

  a->grad += b->data * v->grad;
  b->grad += a->data * v->grad;
}

Value* mul(Value* a, Value* b){
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = new_value(a->data * b->data, children, 2);
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Value* v){
  Value* a = v->_prev[0];
  a->grad += v->exp * pow(a->data, v->exp - 1) * v->grad;
}

Value* pow_val(Value* a, double exp) {
  Value** children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = new_value(pow(a->data, exp), children, 1);
  out->_backward = pow_backward;
  out->exp = exp;
  return out;
}

Value* negate(Value* a) {
  return mul(a, new_value(-1, NULL, 0));
}

Value* sub(Value* a, Value* b) {
  return add(a, negate(b));
}

Value* truediv(Value* a, Value* b) {
  return mul(a, pow_val(b, -1));
}

void print_value(Value* v) {
  printf("Value(data=%.4f, grad=%.4f)\n", v->data, v->grad);
}

void relu_backward(Value* v){
  Value* a = v->_prev[0];
  a->grad += (v->data > 0) * v->grad;
}

Value* relu(Value* a) {
  Value** children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = new_value(a->data > 0 ? a->data : 0, children, 1);
  out->_backward = relu_backward;
}

void build_topo(Value* v, Value*** topo, int* topo_size, Value** visited, int* visited_size) {
  for(int i = 0; i < *visited_size; ++i) {
    if(visited[i] == v) return;
  }
  visited[(*visited_size)++] = v;
  for(int i = 0; i < v->_prev_size; ++i) {
    build_topo(v->_prev[i], topo, topo_size, visited, visited_size);
  }
  (*topo)[(*topo_size)++] = v;
}

void backward(Value *v) {
  int topo_size = 0;
  Value** topo = (Value**)malloc(100 * sizeof(Value*));

  int visited_size = 0;
  Value** visited = (Value**)malloc(100 * sizeof(Value*));

  build_topo(v, &topo, &topo_size, visited, &visited_size);

  v->grad = 1;
  for(int i = topo_size - 1; i >= 0; --i) {
    topo[i]->_backward(topo[i]);
  }

  free(topo);
  free(visited);
}

int main() {
  Value* a = new_value(2.0, NULL, 0);
  Value* b = new_value(3.0, NULL, 0);
  Value* c = add(a, b);
  Value* d = mul(a, b);
  Value* e = relu(c);
  Value* f = pow_val(d, 2);

  backward(f);
  print_value(a);
  print_value(b);
  print_value(c);
  print_value(d);
  print_value(e);
  print_value(f);
  free(a);
  free(b);
  free(c);
  free(d);
  free(e);
  free(f);
  return 0;
}