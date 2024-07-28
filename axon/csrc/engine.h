#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include <string>

class Value {
public:
  double data;
  double grad;
  double exp;
  std::vector<Value*> _prev;
  void (*_backward)(Value*);

  Value(double data);

  static void noop_backward(Value* v);

  static Value* add(Value* a, Value* b);
  static void add_backward(Value* v);

  static Value* mul(Value* a, Value* b);
  static void mul_backward(Value* v);

  static Value* pow_val(Value* a, double exp);
  static void pow_backward(Value* v);

  static Value* negate(Value* a);
  static Value* sub(Value* a, Value* b);
  static Value* truediv(Value* a, Value* b);

  static Value* relu(Value* a);
  static void relu_backward(Value* v);

  static void build_topo(Value* v, std::vector<Value*>& topo, std::vector<Value*>& visited);
  static void backward(Value* v);

  std::string repr() const;
  double get_data() const;
  double get_grad() const;
};

#endif