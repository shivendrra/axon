#include <pybind11/pybind11.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
  py::class_<Value>(m, "Value")
    .def(py::init<double>())
    .def_readwrite("data", &Value::data)
    .def_readwrite("grad", &Value::grad)
    .def("repr", &Value::repr)
    .def("get_data", &Value::get_data)
    .def("get_grad", &Value::get_grad)
    .def_static("add", &Value::add, py::return_value_policy::reference)
    .def_static("mul", &Value::mul, py::return_value_policy::reference)
    .def_static("pow", &Value::pow_val, py::return_value_policy::reference)
    .def_static("negate", &Value::negate, py::return_value_policy::reference)
    .def_static("sub", &Value::sub, py::return_value_policy::reference)
    .def_static("truediv", &Value::truediv, py::return_value_policy::reference)
    .def_static("relu", &Value::relu, py::return_value_policy::reference)
    .def_static("backward", &Value::backward, py::return_value_policy::reference);
}