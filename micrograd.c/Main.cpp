#include "Value.h"
#include "Module.h"
#include <iostream>

int main() {
  MLP mlp(2, {4, 4, 1});
  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0), std::make_shared<Value>(2.0)};
  auto out = mlp(x);
  
  out[0]->grad = 1.0;
  Value::backward(out[0].get());

  for (auto& p : mlp.parameters()) {
    p->print_value();
  }

  return 0;
}