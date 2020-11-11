cmake_minimum_required(VERSION 3.0.0)
project(c++extention VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(c++extention main.cpp)
add_executable(another c++extention/test.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
