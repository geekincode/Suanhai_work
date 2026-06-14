cmake_minimum_required(VERSION 3.18)
project(MatrixProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")

add_executable(matrix_demo src/main.cpp)

include_directories( include)

add_library(matrix_ops SHARED
    src/MatrixCalculations.cpp
)

install(TARGETS matrix_demo
        RUNTIME DESTINATION bin)

install(TARGETS matrix_ops
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include)

