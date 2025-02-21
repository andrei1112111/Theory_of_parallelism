cmake_minimum_required(VERSION 3.22.1)
project(numerical_integration C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-fopenmp -O2")

add_executable(numerical_integration main.c)
target_link_libraries(numerical_integration PRIVATE m)
