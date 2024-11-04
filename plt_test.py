import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
import torch
"""
# 生成示例数据（你可以替换为自己的数据）
np.random.seed(0)
latents = np.random.rand(100, 2)  # 生成100个二维点
"""

latents=torch.load('latent_poit2v_700.pt').detach().numpy()

# 拟合高斯混合模型
gm = GaussianMixture(n_components=3, random_state=0).fit(latents)
print('Average negative log likelihood:', -1 * gm.score(latents))

# 生成二维网格
x = np.linspace(np.min(latents[:, 0]), np.max(latents[:, 0]), 30)
y = np.linspace(np.min(latents[:, 1]), np.max(latents[:, 1]), 30)
X, Y = np.meshgrid(x, y)

# 计算概率密度值
Z_density = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z_density[i, j] = np.exp(gm.score_samples(point.reshape(1, -1)))

# 绘制三维散点图和表面
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(latents[:, 0], latents[:, 1], np.zeros_like(latents[:, 0]), s=5, color='black', label='Data Points')

# 绘制概率密度的表面
ax.plot_surface(X, Y, Z_density, cmap='Reds', alpha=0.5)

# 设置标签和标题
ax.set_title('3D Gaussian Mixture Model of 2D Data')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Density')
ax.legend()

plt.show()
