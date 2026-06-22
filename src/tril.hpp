#ifndef TRIL_HPP
#define TRIL_HPP

#include <vector>
#include <iostream>

// 定义 Matrix 类型为二维向量
using Matrix = std::vector<std::vector<int>>;

// 计算矩阵的下三角部分（包含对角线）
// 上三角部分（对角线以上）置为 0
inline Matrix tril(const Matrix& a) {
    if (a.empty()) return {};
    
    size_t rows = a.size();
    size_t cols = a[0].size();
    
    Matrix result(rows, std::vector<int>(cols, 0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j <= i && j < cols; ++j) {
            result[i][j] = a[i][j];
        }
    }
    
    return result;
}

// 重载 == 运算符用于比较两个 Matrix
inline bool operator==(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

// 重载 << 运算符用于输出 Matrix（gtest 需要）
inline std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "[";
    for (size_t i = 0; i < m.size(); ++i) {
        os << "[";
        for (size_t j = 0; j < m[i].size(); ++j) {
            os << m[i][j];
            if (j + 1 < m[i].size()) os << ", ";
        }
        os << "]";
        if (i + 1 < m.size()) os << ", ";
    }
    os << "]";
    return os;
}

#endif // TRIL_HPP