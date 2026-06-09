from sympy import *  

from sympy.tensor.array import MutableDenseNDimArray
import numpy as np


print("4.1 符号张量创建")
print("=========================================")

x, y = symbols('x, y')  
expr = x**2 + sqrt(y) + Rational(1, 2)

print(expr)


print("\n4.2 爱因斯坦求和约定实现")
print("=========================================")

# 三维符号张量 A (2×2×2)
A = MutableDenseNDimArray(
    [[[symbols(f'a{i}{j}{k}') for k in range(2)]
      for j in range(2)]
     for i in range(2)]
)

# 爱因斯坦求和：对 k 维求和
C = np.einsum('ijk->ij', A)
print(C)





