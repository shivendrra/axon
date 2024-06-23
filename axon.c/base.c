#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "helpers/functionals.h"

#define INT8 0
#define INT16 1
#define INT32 2
#define INT64 3
#define FLOAT16 4
#define FLOAT32 5
#define FLOAT64 6

typedef struct {
  void *data;
  int *shape;
  int ndim;
  int dtype;
} array;

void* allocate_memory(int size, int dtype) {
  switch (dtype) {
    case INT8: return malloc(size * sizeof(char));
    case INT16: return malloc(size * sizeof(short));
    case INT32: return malloc(size * sizeof(int));
    case INT64: return malloc(size * sizeof(long));
    case FLOAT16: return malloc(size * sizeof(float) / 2);
    case FLOAT32: return malloc(size * sizeof(float));
    case FLOAT64: return malloc(size * sizeof(double));
    default: return NULL;
  }
}

void copy_memory(void *dest, void *src, int size, int dtype) {
  switch (dtype) {
    case INT8: memcpy(dest, src, size * sizeof(char)); break;
    case INT16: memcpy(dest, src, size * sizeof(short)); break;
    case INT32: memcpy(dest, src, size * sizeof(int)); break;
    case INT64: memcpy(dest, src, size * sizeof(long)); break;
    case FLOAT16: memcpy(dest, src, size * sizeof(float) / 2); break;
    case FLOAT32: memcpy(dest, src, size * sizeof(float)); break;
    case FLOAT64: memcpy(dest, src, size * sizeof(double)); break;
    default: break;
  }
}

int num_elements(int *shape, int ndim) {
  int num = 1;
  for (int i = 0; i < ndim; i++) {
    num *= shape[i];
  }
  return num;
}

array* array_create(void *data, int *shape, int ndim, int dtype) {
  array *arr = (array *)malloc(sizeof(array));
  arr->shape = (int *)malloc(ndim * sizeof(int));
  arr->ndim = ndim;
  arr->dtype = dtype;
  int numel = num_elements(shape, ndim);
  arr->data = allocate_memory(numel, dtype);
  copy_memory(arr->data, data, numel, dtype);
  memcpy(arr->shape, shape, ndim * sizeof(int));
  return arr;
}

void array_free(array *arr) {
  free(arr->data);
  free(arr->shape);
  free(arr);
}

array* array_copy(array *arr) {
  return array_create(arr->data, arr->shape, arr->ndim, arr->dtype);
}

void array_print(array *arr) {
  int numel = num_elements(arr->shape, arr->ndim);
  if (arr->dtype == FLOAT64) {
    double *data = (double *)arr->data;
    for (int i = 0; i < numel; i++) {
        printf("%f ", data[i]);
    }
  } else if (arr->dtype == INT64) {
    long *data = (long *)arr->data;
    for (int i = 0; i < numel; i++) {
      printf("%ld ", data[i]);
    }
  }
  printf("\n");
}

double array_mean(array *arr) {
  int numel = num_elements(arr->shape, arr->ndim);
  double sum = 0.0;
  if (arr->dtype == FLOAT64) {
    double *data = (double *)arr->data;
    for (int i = 0; i < numel; i++) {
      sum += data[i];
    }
  } else if (arr->dtype == INT64) {
    long *data = (long *)arr->data;
    for (int i = 0; i < numel; i++) {
      sum += data[i];
    }
  }
  return sum / numel;
}

double array_var(array *arr, double mean) {
  int numel = num_elements(arr->shape, arr->ndim);
  double var = 0.0;
  if (arr->dtype == FLOAT64) {
    double *data = (double *)arr->data;
    for (int i = 0; i < numel; i++) {
      var += (data[i] - mean) * (data[i] - mean);
    }
  } else if (arr->dtype == INT64) {
    long *data = (long *)arr->data;
    for (int i = 0; i < numel; i++) {
      var += (data[i] - mean) * (data[i] - mean);
    }
  }
  return var / numel;
}

double array_std(array *arr) {
  double mean = array_mean(arr);
  double var = array_var(arr, mean);
  return sqrt(var);
}

array *tanh(array *arr);

void apply_tanh_recursive(double *data, int *shape, int ndim, int current_dim, double *result) {
  if (current_dim == ndim) {
    *result = tanh_function(*data);
  } else {
    int size = shape[current_dim];
    for (int i = 0; i < size; ++i) {
      apply_tanh_recursive(data + i * shape[current_dim + 1], shape, ndim, current_dim + 1, result + i * shape[current_dim + 1]);
    }
  }
}

array *sigmoid(array *arr) {
  int total_size = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    total_size *= arr->shape[i];
  }

  double *result_data = (double *)malloc(total_size * sizeof(double));
  apply_sigmoid_recursive(arr->data, arr->shape, arr->ndim, 0, result_data);

  array *result = create_array(result_data, arr->shape, arr->ndim);
  return result;
}

array *sigmoid(array *arr);

void apply_sigmoid_recursive(double *data, int *shape, int ndim, int current_dim, double *result) {
  if (current_dim == ndim) {
    *result = tanh_function(*data);
  } else {
    int size = shape[current_dim];
    for (int i = 0; i < size; ++i) {
      apply_sigmoid_recursive(data + i * shape[current_dim + 1], shape, ndim, current_dim + 1, result + i * shape[current_dim + 1]);
    }
  }
}

array *sigmoid(array *arr) {
  int total_size = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    total_size *= arr->shape[i];
  }

  double *result_data = (double *)malloc(total_size * sizeof(double));
  apply_sigmoid_recursive(arr->data, arr->shape, arr->ndim, 0, result_data);

  array *result = create_array(result_data, arr->shape, arr->ndim);
  return result;
}