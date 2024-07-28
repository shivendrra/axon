cmake_minimum_required(VERSION 3.12)
project(engine)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

add_library(engine MODULE axon/csrc/engine.cpp axon/csrc/bindings.cpp)
target_link_libraries(engine PRIVATE pybind11::module)

set_target_properties(engine PROPERTIES PREFIX "" SUFFIX ".pyd")