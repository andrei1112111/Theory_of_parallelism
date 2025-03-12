cmake_minimum_required(VERSION 3.22.1)
project(task2.3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fopenmp -O2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)


add_executable(single single.cpp)

add_executable(static individual.cpp)

add_executable(dynamic individual.cpp)
target_compile_definitions(dynamic PRIVATE SCHEDULE_DYNAMIC)

add_executable(guided individual.cpp)
target_compile_definitions(guided PRIVATE SCHEDULE_GUIDED)

add_executable(auto individual.cpp)
target_compile_definitions(auto PRIVATE SCHEDULE_AUTO)
