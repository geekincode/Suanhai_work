from scipy import linalg
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve

from scipy import integrate
from scipy.optimize import approx_fprime  


print("3.1 线性代数运算（矩阵求逆、特征值、特征向量）")
print("=========================================")

A = np.array([[1, 2], [3, 4]]) 
# 矩阵求逆
A_inv = linalg.inv(A) 
print(A_inv)

# 特征值与特征向量
eigvals, eigvecs = linalg.eig(A)
print('特征值为：\n',eigvecs,'\n','特征向量为：\n',eigvals)  



print("\n3.2 稀疏矩阵（定义、创建、求解）")
print("=========================================")

# 创建稀疏矩阵 (CSR 格式)
data = np.array([1, 2, 3, 4])  # 非零值
row = np.array([0, 0, 1, 2])   # 行索引
col = np.array([0, 2, 1, 2])   # 列索引
sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
print(sparse_mat.toarray())

# 求解 Ax = b (稀疏线性系统)
b = np.array([1, 2, 3])
x = spsolve(sparse_mat, b)  # 求解 x

print(sparse_mat.toarray())  # 转为稠密查看
print(x)


print("\n3.3 数值求导与积分")
print("=========================================")

def f(x):
    return x**2 + np.sin(x)

x0 = np.array([1.0])  # approx_fprime 需要输入数组形式
df = approx_fprime(x0, f, 1e-6)[0]  
# 用步长 1e-6，数值计算函数 f 在点 x0 处的导数。
print(df)

# 定积分 ∫_0^π sin(x) dx
integral, err = integrate.quad(np.sin, 0, np.pi)  
print(integral,'\n',err)