#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ALPHA 0.03

double relu(double x) {
  return fmax(0, x);
}

double relu_derivative(double x) {
  return x > 0 ? 1 : 0;
}

double leaky_relu(double x, double alpha) {
  return x >= 0 ? x : alpha * x;
}

double leaky_relu_derivative(double x, double alpha) {
  return x > 0 ? 1 : alpha;
}

double tanh_derivative(double x) {
  double tanh_x = tanh(x);
  return 1 - tanh_x * tanh_x;
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
  double sig = sigmoid(x);
  return sig * (1 - sig);
}

double cdf(double x) {
  return 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

double pdf(double x) {
  return (1 / sqrt(2 * M_PI)) * exp(-0.5 * x * x);
}

double gelu(double x) {
  return x * cdf(x);
}

double gelu_derivative(double x) {
  return cdf(x) + x * pdf(x);
}