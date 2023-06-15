cmake_minimum_required(VERSION 3.25)
project(CS205_Project2)

set(CMAKE_CXX_STANDARD 17)

add_executable(CS205_Project2 main.cpp project.h)
add_executable(CleanupBCData preprocess.cpp)
