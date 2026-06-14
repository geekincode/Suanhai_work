

``` bash
g++ src/main.cpp src/MatrixCalculations.cpp \
    -Iinclude -I/usr/include/eigen3 \
    -o bin/matrix_demo
./bin/matrix_demo
```


## Eigen 的广播与按维度运算

### 核心差异

| 特性 | NumPy/PyTorch | Eigen |
|------|--------------|-------|
| 广播 | **自动推断**，隐式扩展维度 | **显式声明**，通过 `rowwise()` / `colwise()` |
| 语法 | `A + b` 自动广播 | `A.rowwise() + b` 必须指明方向 |
| 惰性求值 | 部分支持 | 全面支持，表达式模板零拷贝 |

### 关键机制：`rowwise()` / `colwise()`

Eigen 没有全局自动广播，而是要求你显式指定**沿哪个维度**操作：

```cpp
#include <Eigen/Core> #include <iostream/>

int main() { Eigen::Matrix3d A = Eigen::Matrix3d::Random(); Eigen::Vector3d b = Eigen::Vector3d::Random();
// ❌ 错误：Eigen 不会自动广播
// auto C = A + b;  
// ✅ 正确：显式声明按行广播（每行都加 b）
auto C = A.rowwise() + b.transpose();  // b 是列向量，需转置为行向量
// ✅ 按列广播：每列减去列向量均值
Eigen::Vector3d col_mean = A.colwise().mean();
auto A_centered = A.colwise() - col_mean;


## 题 4：广播机制简要探究

简要说明 Eigen 如何实现类似 NumPy/PyTorch 的广播或按行/按列运算。

---

### 核心概念

Eigen 的广播机制与 NumPy/PyTorch 的**自动广播**有本质差异：

| 特性 | NumPy/PyTorch | Eigen |
|------|---------------|-------|
| 广播方式 | 自动隐式广播（按维度对齐） | **显式**指定方向（`rowwise()`/`colwise()`） |
| 语法 | `A + v` 自动判断 | `A.rowwise() + v` 必须显式写出 |
| 灵活性 | 高（自动扩展任意维度） | 受限（主要支持行/列方向） |

---

### 1. `rowwise()` 与 `colwise()`

Eigen 通过这两个函数显式声明广播方向：

```cpp
#include &lt;Eigen/Dense&gt;
#include &lt;iostream&gt;

int main() {
    Eigen::Matrix3d A;
    A &lt;&lt; 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    Eigen::Vector3d v(10, 20, 30);

    // 按行广播：v 被视为行向量，加到每一行
    Eigen::Matrix3d row_result = A.rowwise() + v.transpose();
    std::cout &lt;&lt; "A.rowwise() + v^T:\n" &lt;&lt; row_result &lt;&lt; "\n\n";
    // 结果：每一行都加上 [10, 20, 30]
    // [[11, 22, 33],
    //  [14, 25, 36],
    //  [17, 28, 39]]

    // 按列广播：v 被视为列向量，加到每一列
    Eigen::Matrix3d col_result = A.colwise() + v;
    std::cout &lt;&lt; "A.colwise() + v:\n" &lt;&lt; col_result &lt;&lt; "\n\n";
    // 结果：每一列都加上 [10, 20, 30]^T
    // [[11, 12, 13],
    //  [24, 25, 26],
    //  [37, 38, 39]]

    // 行/列归约操作
    Eigen::Vector3d row_sum = A.rowwise().sum();  // 每行求和 → [6, 15, 24]
    Eigen::Vector3d col_sum = A.colwise().sum();  // 每列求和 → [12, 15, 18]

    return 0;
}