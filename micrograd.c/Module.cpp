#include "Module.h"

void Module::zero_grad() {
  for (auto& p : parameters()) {
    p->grad = 0;
  }
}

std::vector<std::shared_ptr<Value>> Module::parameters() {
  return {};
}

Neuron::Neuron(int nin, bool nonlin)
  : nonlin(nonlin) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1, 1);

  for (int i = 0; i < nin; ++i) {
    w.push_back(std::make_shared<Value>(dis(gen)));
  }
  b = std::make_shared<Value>(0.0);
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  auto act = std::accumulate(x.begin(), x.end(), b, 
    [this](std::shared_ptr<Value> sum, std::shared_ptr<Value> xi) {
      return Value::add(sum, Value::mul(this->w[&xi - &x[0]], xi));
    }
  );
  return nonlin ? Value::relu(act) : act;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
  auto params = w;
  params.push_back(b);
  return params;
}

std::string Neuron::repr() const {
  std::ostringstream oss;
  oss << (nonlin ? "ReLU" : "Linear") << "Neuron(" << w.size() << ")";
  return oss.str();
}

Layer::Layer(int nin, int nout, bool nonlin) {
  for (int i = 0; i < nout; ++i) {
    neurons.push_back(std::make_shared<Neuron>(nin, nonlin));
  }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out;
  for (auto& n : neurons) {
    out.push_back((*n)(x));
  }
  return out.size() == 1 ? std::vector<std::shared_ptr<Value>>{out[0]} : out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& n : neurons) {
    auto n_params = n->parameters();
    params.insert(params.end(), n_params.begin(), n_params.end());
  }
  return params;
}

std::string Layer::repr() const {
  std::ostringstream oss;
  oss << "Layer of [";
  for (size_t i = 0; i < neurons.size(); ++i) {
    oss << neurons[i]->repr();
    if (i < neurons.size() - 1) oss << ", ";
  }
  oss << "]";
  return oss.str();
}

MLP::MLP(int nin, const std::vector<int>& nouts) {
  std::vector<int> sz = {nin};
  sz.insert(sz.end(), nouts.begin(), nouts.end());
  for (size_t i = 0; i < nouts.size(); ++i) {
    layers.push_back(std::make_shared<Layer>(sz[i], sz[i + 1], i != nouts.size() - 1));
  }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out = x;
  for (auto& layer : layers) {
    out = (*layer)(out);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& layer : layers) {
    auto l_params = layer->parameters();
    params.insert(params.end(), l_params.begin(), l_params.end());
  }
  return params;
}

std::string MLP::repr() const {
  std::ostringstream oss;
  oss << "MLP of [";
  for (size_t i = 0; i < layers.size(); ++i) {
    oss << layers[i]->repr();
    if (i < layers.size() - 1) oss << ", ";
  }
  oss << "]";
  return oss.str();
}