cmake_minimum_required(VERSION 3.22.1)
project(task3.2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)


FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_executable(main main.cpp)
TARGET_LINK_LIBRARIES(main ${Boost_LIBRARIES})
