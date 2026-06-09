import matplotlib.pyplot as plt
import numpy as np


print("5.2 折线图绘制")
print("=========================================")

x = np.linspace(0, 2 * np.pi, 100) # 生成 0 到 2π 之间的 100 个点
y_sin = np.sin(x)
y_cos = np.cos(x)

# 2. 创建 Figure 和 Axes
fig, ax = plt.subplots() 
# 3. 绘图
plt.plot(x, y_sin, label='sin(x)') # 绘制 sin(x) 曲线
plt.plot(x, y_cos, label='cos(x)', linestyle='--') # 绘制 cos(x) 曲线，使用虚线
# 4. 添加图表元素
plt.title("Figure") # 设置图表标题
plt.xlabel("X")     # 设置 X 轴标签
plt.ylabel("Y")     # 设置 Y 轴标签
plt.legend()        # 显示图例 
plt.grid(True)      # 显示网格
# 5. 显示图表
# plt.show()


print("5.3 热图可视化张量")
print("=========================================")

mat = np.random.rand(5, 5)
plt.imshow(mat, cmap='viridis')
plt.colorbar()
plt.title('Tensor Heatmap')
# plt.show()


print("5.4 向量场可视化")
print("=========================================")

X, Y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
U = -Y  # 示例梯度场
V = X

plt.quiver(X, Y, U, V)
plt.title('Gradient Field')
# plt.show()




