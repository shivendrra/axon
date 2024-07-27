#include <pybind11/pybind11.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
  py::class_<Value>(m, "Value")
    .def(py::init<double>())
    .def_readwrite("data", &Value::data)
    .def_readwrite("grad", &Value::grad)
    .def("print_value", &Value::print_value)
    .def_static("add", &Value::add, py::return_value_policy::reference)
    .def_static("mul", &Value::mul, py::return_value_policy::reference)
    .def_static("pow_val", &Value::pow_val, py::return_value_policy::reference)
    .def_static("negate", &Value::negate, py::return_value_policy::reference)
    .def_static("sub", &Value::sub, py::return_value_policy::reference)
    .def_static("truediv", &Value::truediv, py::return_value_policy::reference)
    .def_static("relu", &Value::relu, py::return_value_policy::reference)
    .def_static("backward", &Value::backward, py::return_value_policy::reference);
}