cmake_minimum_required(VERSION 3.12)
project(engine)

set(CMAKE_CXX_STANDARD 14)

set(pybind11_DIR "C:/Users/shivh/AppData/Roaming/Python/Python311/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

set(CMAKE_C_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe")

add_library(engine MODULE axon/csrc/engine.cpp axon/csrc/bindings.cpp)
target_link_libraries(engine PRIVATE pybind11::module)

set_target_properties(engine PROPERTIES PREFIX "" SUFFIX ".pyd")