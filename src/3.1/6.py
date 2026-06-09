import numpy as np
import scipy.integrate  
from sympy import symbols, integrate, sin, simplify
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 指定默认字体：文泉驿正黑
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号无法显示的问题

# 函数1: SymPy - 符号积分（输入：函数表达式、下限、上限，输出：符号积分值）
def sympy_integral(u_expr, lower, upper):
    x = symbols('x')
    integral = integrate(u_expr, (x, lower, upper))
    return simplify(integral)

# 函数2: NumPy - 生成网格与函数值（输入：点数、下限、上限、函数，输出：x 值、u 值）
def numpy_grid(n, lower, upper, func):
    x_vals = np.linspace(lower, upper, n)
    u_vals = func(x_vals)
    return x_vals, u_vals

# 函数3: SciPy - 数值积分（输入：x_vals、u_vals，输出：数值积分值）
def scipy_numerical_integral(x_vals, u_vals):
    integral_num = scipy.integrate.simpson(u_vals, x_vals)  # Fully qualified
    return integral_num

# 函数4: Matplotlib - 可视化函数与积分区域（输入：x_vals、u_vals，输出：显示图表）
def plot_function_with_area(x_vals, u_vals):
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, u_vals, label='u = sin(x)', color='blue')
    plt.fill_between(x_vals, u_vals, color='skyblue', alpha=0.4, label='积分区域')
    plt.title('函数 u = sin(x) 与积分区域')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

# 实例化演示：运行整个流程
n = 100          # 小规模点数，便于理解
lower, upper = 0, np.pi  # 积分区间
func = np.sin   # 数值函数

# SymPy 实例：符号积分
u_expr = sin(symbols('x'))
symbolic_int = sympy_integral(u_expr, lower, upper)
print("符号积分值:", float(symbolic_int))  # 2.0

# NumPy 实例：网格生成
x_vals, u_vals = numpy_grid(n, lower, upper, func)

# SciPy 实例：数值积分
numerical_int = scipy_numerical_integral(x_vals, u_vals)
print("数值积分值:", numerical_int)  # 约 2.0

# Matplotlib 实例：可视化
plot_function_with_area(x_vals, u_vals)