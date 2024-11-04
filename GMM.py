from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal


def draw_ellipse(mean, cov, ax=None, **kwargs):
    """绘制由均值和协方差矩阵决定的椭圆"""
    ax = ax or plt.gca()

    # 计算协方差矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 按特征值大小排序
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 计算椭圆的角度，特征向量的方向
    vx, vy = eigvecs[:, 0]
    theta = np.degrees(np.arctan2(vy, vx))

    # 椭圆的宽度和高度与特征值成正比
    width, height = 2 * np.sqrt(eigvals)

    # 使用 matplotlib 的 Ellipse 类绘制椭圆
    ellipse = Ellipse(mean, width, height, angle=theta, **kwargs)
    ax.add_patch(ellipse)


"""
# 生成模拟数据
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 2, 300).reshape(-1, 1),
                    np.random.normal(-2, 1, 300).reshape(-1, 1),
                    np.random.normal(3, 0.5, 300).reshape(-1, 1)])
"""
"""
latent_poit = torch.load('latent_poit2v.pt')
data=latent_poit.detach().numpy()
# 初始化高斯混合模型
gmm = GaussianMixture(n_components=3, random_state=42)

# 训练模型
gmm.fit(data)

# 预测数据点属于哪个高斯分布
labels = gmm.predict(data)

# 输出每个高斯分布的参数
print("Means of Gaussians:")
print(gmm.means_)
print("Covariances of Gaussians:")
print(gmm.covariances_)

#  可视化聚类结果
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('GMM Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
"""
#加载在训练完成后保存的tensor张量

latents=torch.load('latent_poit2v_700.pt').detach().numpy()

gm = GaussianMixture(n_components=3, random_state=0, init_params='kmeans').fit(
    latents)  # plot a n_components v.s. Average negative log likelihood

#数据越小，高斯混合分布得拟合效果越好（调整超惨n_components=5）
print('Average negative log likelihood:', -1 * gm.score(latents))

# 生成二维网格
x = np.linspace(np.min(latents[:, 0]), np.max(latents[:, 0]), 500)
y = np.linspace(np.min(latents[:, 1]), np.max(latents[:, 1]), 500)
X, Y = np.meshgrid(x, y)
XY = np.array([X.ravel(), Y.ravel()]).T

# 计算网格上每个点的概率密度值
Z = np.exp(gm.score_samples(XY))  # score_samples返回的是log likelihood，需要exp转换为概率
Z = Z.reshape(X.shape)

# 绘制填充的等高线
plt.figure(figsize=(8, 6))
contour_filled = plt.contourf(X, Y, Z, levels=10, cmap='Reds', alpha=0.7)  # 使用 'Reds' 色图进行填充
plt.colorbar(contour_filled, label='Density')  # 显示颜色条

# 绘制数据点
plt.scatter(latents[:, 0], latents[:, 1], s=5, color='black', label='Data Points',alpha=0.2)  # 数据点用黑色标记
plt.title('Gaussian Mixture Model Contour Plot with Red Fill')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

plt.show()


def adjust_plot_limits(gm, ax):
    """根据高斯混合模型调整绘图的坐标范围"""
    means = gm.means_
    covariances = gm.covariances_

    # 计算每个成分的边界（均值加减若干标准差）
    margin = 3  # 偏移多个标准差，通常取3倍标准差
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for mean, cov in zip(means, covariances):
        eigvals, _ = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(eigvals)
        x_min = min(x_min, mean[0] - margin * width)
        x_max = max(x_max, mean[0] + margin * width)
        y_min = min(y_min, mean[1] - margin * height)
        y_max = max(y_max, mean[1] + margin * height)

    # 设置坐标范围
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


# 示例调用
fig, ax = plt.subplots()
plt.figure(figsize=(6, 6), dpi=100)
plt.scatter(latents[:, 0], latents[:, 1], s=10, alpha=0.7, label='Latent points')
for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
    draw_ellipse(pos, covar, ax=ax, alpha=0.75 * w * 2, facecolor='red', zorder=-10)

# 调整坐标轴范围
adjust_plot_limits(gm, ax)

plt.show()

#定义分布方便MCMC调用
"""
def gm():
    gm = GaussianMixture(n_components=1, random_state=0, init_params='kmeans').fit(
        latents)
    return gm
"""
w_factor = 0.2 / gm.weights_.max()
for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
    print(pos)
    print(covar)
    print(w)
    draw_ellipse(pos, covar, alpha=0.75 * w * w_factor, facecolor='slategrey', zorder=-10)
plt.show()

"""
# 3. 绘制散点图
plt.figure(figsize=(6, 6), dpi=100)
plt.scatter(latents[:, 0], latents[:, 1], s=10, alpha=0.7, label='Latent points')
# 4. 绘制 GMM 拟合的高斯分布的等高线
mean = gm.means_[0]  # GMM 的均值
covariance = gm.covariances_[0]  # GMM 的协方差矩阵

# 创建网格点
x, y = np.mgrid[latents[:, 0].min():latents[:, 0].max():.01, latents[:, 1].min():latents[:, 1].max():.01]
pos = np.dstack((x, y))

# 创建二维高斯分布
rv = multivariate_normal(mean, covariance)

# 计算每个网格点的概率密度
pdf_values = rv.pdf(pos)

# 绘制填充等高线图，颜色表示概率深浅
contour = plt.contourf(x, y, pdf_values, levels=50, cmap='Reds',alpha=0.65)

# 添加颜色条，展示概率密度的深浅
plt.colorbar(contour)

# 绘制等高线图
plt.contour(x, y, rv.pdf(pos), cmap='plasma',alpha=0.1)

# 设置图形标题和标签
plt.title('Gaussian Mixture Model - 1 Component')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()
plt.show()
"""



"""
plot_gmm(gm, latents)

# Using elbow method to find out the best # of components, the lower the negative log likehood the better the model is, but too many cluster is trivial. just imagine you fit each individual data points with a Gaussian, in this case, you would have a very good model. but the such fitting is not very useful.

# In this case, the best number of cluster is either 4 or 5.

scores = []  # using elbow method to find out the best # of components
for i in range(1, 8):
    gm = GaussianMixture(n_components=i, random_state=0, init_params='kmeans').fit(latents)
    print('Average negative log likelihood:', -1 * gm.score(latents))
    scores.append(-1 * gm.score(latents))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure()
plt.scatter(range(1, 8), scores, color='green')
plt.plot(range(1, 8), scores)
plt.savefig('elbow_plot.png', format='png', dpi=300)
plt.show()
"""
