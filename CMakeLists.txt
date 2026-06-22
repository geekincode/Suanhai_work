cmake_minimum_required(VERSION 3.14)
project(tril_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 GTest
find_package(GTest REQUIRED)

# 添加测试可执行文件
add_executable(test_tril src/test_tril.cpp)

# 链接 GTest 库
target_link_libraries(test_tril GTest::gtest GTest::gtest_main)

# 添加测试
enable_testing()
add_test(NAME TrilTest COMMAND test_tril)