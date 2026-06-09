import numpy as np

print("2.1 张量创建、形状")
print("=========================================")

# 向量 (1D 张量)
vec = np.array([1, 2, 3])  # 形状 (3,)
print(vec.shape)  # (3,)

# 矩阵 (2D 张量)
mat = np.array([[1, 2], [3, 4], [5, 6]])  # 形状 (3, 2)
print(mat.shape)  # (3, 2)

# 高阶张量 (3D)
tensor3d = np.random.rand(2, 3, 4)  # 形状 (2, 3, 4)
print(tensor3d.shape)  # (2, 3, 4)


print("\n2.2 张量基本运算（加减乘除）")
print("=========================================")

A = np.arange(0, 10, 3).reshape(-1,1)  # (4，1)
print(A)
B = np.array([5, 6])  # (2,)

C_add = A + B 
C_sub = A - B  
C_mul = A * B  
C_div = A / B 

print(C_add)
print(C_sub)
print(C_mul)
print(C_div)

x = np.arange(12).reshape(4, 3)
print(x)
print(type(x))
print(x.dtype)
print(x.shape)   # (4, 3)
print(x.strides) # (24, 8)


print("\n2.3 内积（点积）")
print("=========================================")

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
dot_vec = np.dot(vec1, vec2)  # 1*4 + 2*5 + 3*6 = 32

mat1 = np.array([[1, 2], [3, 4]])  # (2, 2)
mat2 = np.array([[5, 6], [7, 8]])  # (2, 2)
dot_mat = np.dot(mat1, mat2)  # 或 mat1 @ mat2
# [[19, 22], [43, 50]]

print(dot_mat)


print("\n2.4 缩并")
print("=========================================")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])

# 手动循环实现缩并（严格对应 ∑_j A[i,j] * B[j,l]）
C_manual = np.zeros((3, 3))
for i in range(3):
    for l in range(3):
        for j in range(3):
            C_manual[i, l] += A[i, j] * B[j, l]

print(C_manual)

C_matmul = A @ B 
print(C_matmul)
print(np.allclose(C_manual,C_matmul))


print("\n2.5 爱因斯坦求和")
print("=========================================")

A = np.array([[1, 2], [3, 4]])  # 矩阵 A (2x2)
B = np.array([[5, 6], [7, 8]])  # 矩阵 B (2x2)

# 使用 einsum 实现矩阵乘法 (A @ B)
result = np.einsum('ij,jk->ik', A, B)
print("结果 (A @ B):\n", result)


