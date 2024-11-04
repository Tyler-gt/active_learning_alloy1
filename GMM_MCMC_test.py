import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 生成一些数据
np.random.seed(0)
data = np.concatenate([
    np.random.normal(0, 1, 500),  # 第一个高斯成分
    np.random.normal(5, 1, 500)   # 第二个高斯成分
])

# 设定高斯混合模型参数
K = 2  # 高斯成分数量
mu = np.random.rand(K) * 10  # 均值
sigma = np.ones(K)  # 方差
pi = np.array([0.5, 0.5])  # 权重

# MCMC步骤（示例）
n_iterations = 1000
samples = []

for _ in range(n_iterations):
    # 根据当前参数生成新的样本
    component = np.random.choice(K, p=pi)  # 根据权重选择高斯成分
    sample = np.random.normal(mu[component], sigma[component])
    samples.append(sample)

# 绘制结果
plt.hist(samples, bins=30, density=True, alpha=0.5, color='g')
x = np.linspace(-2, 10, 1000)
plt.plot(x, pi[0] * norm.pdf(x, mu[0], sigma[0]), 'r', label='GMM Component 1')
plt.plot(x, pi[1] * norm.pdf(x, mu[1], sigma[1]), 'b', label='GMM Component 2')
plt.title('GMM using MCMC Sampling')
plt.legend()
plt.show()
