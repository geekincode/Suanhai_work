"""
综合作业：NumPy 张量运算、SciPy 稀疏矩阵求解和数值积分可视化
包含：
1. NumPy 张量运算（加减、内积、缩并）和 Matplotlib 可视化
2. SciPy 稀疏矩阵的构造和稀疏线性方程组的求解
3. 综合案例：数值积分的类封装和改进可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import integrate
from matplotlib.patches import Polygon
from matplotlib import rcParams

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 指定默认字体：文泉驿正黑
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号无法显示的问题


# ==============================================================================
# 第一部分：NumPy 张量运算与 Matplotlib 可视化
# ==============================================================================
print("=" * 80)
print("第一部分：NumPy 张量运算（加减、内积、缩并）和 Matplotlib 可视化")
print("=" * 80)

# 创建 3D 张量 T，形状为 (2, 3, 4)
T = np.arange(24).reshape(2, 3, 4)
print(f"\n3D 张量 T (形状: {T.shape}):")
print(T)

# 构造权重向量
w = np.array([1, 2, -1, 0.5])
print(f"\n权重向量 w: {w}")

# 使用 einsum 进行加权求和：沿最后一个轴的加权求和
M = np.einsum('ijk,k->ij', T, w)
print(f"\n加权求和结果 M (形状: {M.shape}):")
print(M)

# 验证计算（手工计算验证）
# print("\n验证计算（以 M[0,0] 为例）:")
# print(f"M[0,0] = T[0,0,0]*w[0] + T[0,0,1]*w[1] + T[0,0,2]*w[2] + T[0,0,3]*w[3]")
# print(f"      = {T[0,0,0]}*{w[0]} + {T[0,0,1]}*{w[1]} + {T[0,0,2]}*{w[2]} + {T[0,0,3]}*{w[3]}")
# manual_result = T[0,0,0]*w[0] + T[0,0,1]*w[1] + T[0,0,2]*w[2] + T[0,0,3]*w[3]
# print(f"      = {manual_result}（程序输出：{M[0,0]}）")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 原始张量的两个切片
ax = axes[0]
im0 = ax.imshow(T[0], cmap='RdBu_r', aspect='auto')
ax.set_title('张量 T[0]（第一个切片）')
ax.set_xlabel('维度 k')
ax.set_ylabel('维度 j')
plt.colorbar(im0, ax=ax)

ax = axes[1]
im1 = ax.imshow(T[1], cmap='RdBu_r', aspect='auto')
ax.set_title('张量 T[1]（第二个切片）')
ax.set_xlabel('维度 k')
ax.set_ylabel('维度 j')
plt.colorbar(im1, ax=ax)

# 加权求和结果
ax = axes[2]
im2 = ax.imshow(M, cmap='RdBu_r', aspect='auto')
ax.set_title('加权求和结果 M = einsum(\'ijk,k->ij\', T, w)')
ax.set_xlabel('维度 j')
ax.set_ylabel('维度 i')
plt.colorbar(im2, ax=ax)

# 添加文本注释
ax.text(0.5, -0.15, f'权重向量: w = {w}', 
        ha='center', transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('./tensor_operations.png', dpi=100, bbox_inches='tight')
print("\n张量运算可视化已保存到 ./tensor_operations.png")
plt.close()


# ==============================================================================
# 第二部分：SciPy 稀疏矩阵求解（n=50）
# ==============================================================================
print("\n" + "=" * 80)
print("第二部分：SciPy 稀疏矩阵求解")
print("=" * 80)

n = 50
h = 1 / (n + 1)
print(f"\n参数设置：n = {n}，步长 h = {h:.6f}")

# 坐标向量
x = np.array([i * h for i in range(1, n + 1)])
print(f"坐标向量 x_i 的范围：[{x[0]:.6f}, {x[-1]:.6f}]")

# 构造三对角稀疏矩阵 A
# 主对角线: 2/h^2, 上/下对角线: -1/h^2
main_diag = np.full(n, 2.0 / (h**2))
off_diag = np.full(n - 1, -1.0 / (h**2))

A = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n), format='csr')
print(A.toarray())
print(f"\n三对角矩阵 A 的形状：{A.shape}，非零元素个数：{A.nnz}")

# 构造右端向量 b
# 函数形状为 b_i = x_i(1-x_i)，在 [0, 1] 上
b = x * (1 - x)
print(f"右端向量 b_i = x_i(1-x_i)，范围：[{b.min():.6f}, {b.max():.6f}]")

# 求解线性方程组
u = spsolve(A, b)
print(f"解向量 u 的范围：[{u.min():.6f}, {u.max():.6f}]")

# 绘制结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 解的折线图
ax = axes[0]
ax.plot(x, u, 'b-o', linewidth=2, markersize=4, label='数值解 u(x)')
ax.set_xlabel('坐标 x')
ax.set_ylabel('解 u(x)')
ax.set_title(f'稀疏线性方程组求解（n={n}）')
ax.grid(True, alpha=0.3)
ax.legend()

# 右端向量的可视化
ax = axes[1]
ax.bar(range(1, n + 1), b, width=1, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('索引 i')
ax.set_ylabel(r'$b_i = x_i(1-x_i)$')
ax.set_title('右端向量 b')
ax.set_xlim(0, n + 1)

plt.tight_layout()
plt.savefig('./sparse_solve_n50.png', dpi=100, bbox_inches='tight')
print("稀疏求解可视化已保存到 ./sparse_solve_n50.png")
plt.close()


# ==============================================================================
# 第二部分补充：小规模情形（n=3）的手工计算和对比
# ==============================================================================
print("\n" + "-" * 80)
print("小规模情形（n=3）的手工计算和对比")
print("-" * 80)

n_small = 3
h_small = 1 / (n_small + 1)
print(f"\n小规模参数：n = {n_small}，h = {h_small:.6f}")

# 坐标
x_small = np.array([i * h_small for i in range(1, n_small + 1)])
print(f"坐标：x = {x_small}")

# 矩阵系数
coeff = 1 / (h_small**2)
print(f"\n系数 1/h² = {coeff:.1f}")

# 构造矩阵 A（3×3 三对角矩阵）
A_small = sparse.diags(
    [np.full(n_small - 1, -coeff), np.full(n_small, 2 * coeff), np.full(n_small - 1, -coeff)],
    [-1, 0, 1],
    shape=(n_small, n_small),
    format='csr'
)

print("\n矩阵 A:")
print(A_small.toarray())

# 右端向量
b_small = x_small * (1 - x_small)
print(f"\n右端向量 b：{b_small}")

# 求解
u_small = spsolve(A_small, b_small)
print(f"解向量 u：{u_small}")

# 手工计算验证（仅为说明，这里用数值方法）
print("\n验证：计算 A*u - b")
residual = A_small.dot(u_small) - b_small
print(f"残差（应接近0）：{residual}")
print(f"残差范数：{np.linalg.norm(residual):.2e}")


# ==============================================================================
# 第三部分：综合案例 - 数值积分的类封装
# ==============================================================================
print("\n" + "=" * 80)
print("第三部分：综合案例 - 数值积分的类封装和改进可视化")
print("=" * 80)


class NumericalIntegral:
    """数值积分的封装类"""
    
    def __init__(self, func, a, b, func_name="函数"):
        """
        初始化数值积分对象
        
        参数：
            func: 被积函数
            a: 积分下界
            b: 积分上界
            func_name: 函数名称（用于显示）
        """
        self.func = func
        self.a = a
        self.b = b
        self.func_name = func_name
        self.integral_value = None
        self.error = None
        
    def compute(self):
        """使用 SciPy 的 quad 方法计算数值积分"""
        self.integral_value, self.error = integrate.quad(self.func, self.a, self.b)
        return self.integral_value, self.error
    
    def plot(self, figsize=(12, 5)):
        """绘制函数曲线，填充正值区域，标注积分值和误差"""
        
        if self.integral_value is None:
            self.compute()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 生成采样点
        x = np.linspace(self.a, self.b, 500)
        y = self.func(x)
        
        # 第一个图：完整曲线和填充区域
        ax = ax1
        ax.plot(x, y, 'b-', linewidth=2.5, label=self.func_name)
        ax.axhline(y=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.3, color='green', 
                        label='正值区域')
        ax.fill_between(x, 0, y, where=(y < 0), alpha=0.3, color='red', 
                        label='负值区域')
        
        # 标注积分值和误差
        text_str = (f'数值积分值: {self.integral_value:.8f}\n'
                   f'估计误差: {self.error:.2e}')
        ax.text(0.5, 0.95, text_str, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'{self.func_name}的数值积分\n区间: [{self.a:.2f}, {self.b:.2f}]',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(self.a, self.b)
        
        # 第二个图：仅填充正值区域的放大视图
        ax = ax2
        positive_mask = y >= 0
        x_pos = x[positive_mask]
        y_pos = y[positive_mask]
        
        if len(x_pos) > 0:
            ax.fill_between(x_pos, 0, y_pos, alpha=0.4, color='green',
                           label='填充区域', edgecolor='darkgreen', linewidth=1.5)
            ax.plot(x_pos, y_pos, 'g-', linewidth=2.5)
        
        # 绘制完整曲线（淡色）
        ax.plot(x, y, 'b--', linewidth=1.5, alpha=0.5, label='完整曲线')
        ax.axhline(y=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('正值区域的详细视图', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(self.a, self.b)
        
        plt.tight_layout()
        return fig


# 定义新的目标函数：u(x) = exp(-x) * cos(2x)
def target_func(x):
    """
    目标函数：u(x) = exp(-x) * cos(2x)
    定义在 [0, 2π] 上
    
    该函数是可积的，因为 exp(-x) 快速衰减，cos(2x) 有界
    """
    return np.exp(-x) * np.cos(2 * x)


# 创建积分对象并计算
print("\n新的目标函数：u(x) = exp(-x) * cos(2x)")
print("积分区间：[0, 2π]")
print("-" * 40)

integrator = NumericalIntegral(
    func=target_func,
    a=0,
    b=2 * np.pi,
    func_name=r'$u(x) = e^{-x}\cos(2x)$'
)

integral_value, error = integrator.compute()
print(f"\n计算结果：")
print(f"  数值积分值：{integral_value:.10f}")
print(f"  估计误差：{error:.2e}")

# 绘制可视化
fig = integrator.plot(figsize=(14, 5))
plt.savefig('./numerical_integral.png', dpi=100, bbox_inches='tight')
print("\n积分可视化已保存到 ./numerical_integral.png")
plt.close()

# 验证：手工分析该积分
print("\n理论分析：")
print("对于 u(x) = exp(-x)*cos(2x)，可以用分部积分或复数方法求原函数。")
print("原函数形式为：F(x) = exp(-x)*(-sin(2x) - 2*cos(2x))/5 + C")
print("\n计算定积分值：")
def F(x):
    """原函数"""
    return np.exp(-x) * (-np.sin(2*x) - 2*np.cos(2*x)) / 5

theoretical_value = F(2 * np.pi) - F(0)
print(f"  理论值（用原函数）：{theoretical_value:.10f}")
print(f"  数值积分值：     {integral_value:.10f}")
print(f"  绝对误差：       {abs(theoretical_value - integral_value):.2e}")


# ==============================================================================
# 额外的对比演示：使用不同的目标函数
# ==============================================================================
print("\n" + "=" * 80)
print("额外演示：对比不同的目标函数")
print("=" * 80)

# 演示 1：原始的 sin(x)（课堂示例）
print("\n演示 1：原始的 sin(x)")
print("-" * 40)
integrator_sin = NumericalIntegral(
    func=np.sin,
    a=0,
    b=np.pi,
    func_name=r'$u(x) = \sin(x)$'
)
val_sin, err_sin = integrator_sin.compute()
print(f"∫₀^π sin(x)dx = {val_sin:.10f}")
print(f"理论值应为 2（cos(π) - cos(0) = -1 - 1 = -2，取绝对值）= 2.0")

# 演示 2：新函数的对比
print("\n演示 2：新函数 exp(-x)*cos(2x)")
print("-" * 40)
print(f"∫₀^(2π) exp(-x)*cos(2x)dx = {integral_value:.10f}")

print("\n" + "=" * 80)
print("所有计算已完成！")
print("=" * 80)
