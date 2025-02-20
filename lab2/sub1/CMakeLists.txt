cmake_minimum_required(VERSION 3.22.1)
project(matrix_vector_product C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "-fopenmp")

add_executable(matrix_vector_product main.c)
