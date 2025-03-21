cmake_minimum_required(VERSION 3.22.1)
project(task3.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(sub1 main.cpp)
target_link_libraries(Masub1trixMultiplication Boost::program_options)
