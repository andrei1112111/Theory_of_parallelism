cmake_minimum_required(VERSION 3.30)
project(my_program)
set(CMAKE_CXX_STANDARD 17)

# ~~~   USE DOUBLE. TO USE FLOAT REMOVE IT   ~~~
add_compile_definitions(USE_DOUBLE)

add_executable(my_program main.cpp)
