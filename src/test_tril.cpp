#include <gtest/gtest.h>
#include "tril.hpp"

TEST(TrilTest, Basic) {
    Matrix a{{1,2,3},{4,5,6},{7,8,9}}; // 定义矩阵 a
    Matrix exp{{1,0,0},{4,5,0},{7,8,9}}; // 定义预期结果 exp
    Matrix res = tril(a); // 计算结果
    EXPECT_EQ(res, exp); // 逐个元素比较
}

