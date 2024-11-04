import numpy as np
from scipy.special import gamma
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from GMM import gm
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
"""
def MCMC(P,X0,chain,space):
    #P=随机变量X服从的概率密度函数
    #X0=MCMC的初始值
    #chain=MCMC的长度,一般来说要生成1000以上的马氏链
    #space=随机变量的取值范围，如[0,inf]

    if not callable(P):
        raise Exception("P必须是函数")

    X_current=X0
    X=[X_current]  #生成的链存储在X的链表里

    while True:
        Delta_X=scipy.stats.norm(loc=0,scale=2).rvs()  #生成一个服从标准差为0，正态分布为2的随机数Delta_X
        X_proposed=X_current+Delta_X

        if X_proposed<space[0] or X_proposed>space[1]:
            p_moving =0
        elif P(X_current)==0:
            p_moving=1
        else:
            p_moving =min(1,P(X_proposed)/P(X_current))

        if scipy.stats.uniform().rvs()<=p_moving:
            X.append(X_proposed)
            X_current=X_proposed
        else:
            X.append(X_current)

        if len(X)>=chain:
            break

    return np.array(X)

def GammaDist(x,k,theta):
    return 1/(gamma(k)*theta**k)*x**(k-1)*np.exp(-x/theta)

gammadist=lambda x:GammaDist(x,5,1)
X=MCMC(gammadist,2,1000,[0,np.inf])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
fig.suptitle('MCMC process')
def anima_chain(index):
    ax1.clear()

    ax1.plot(X[:index+1],'r-')
    ax1.set_xlabel('chain')
    ax1.set_ylabel('X')

def anima_density(index):
    ax2.clear()

    x=np.arange(0.,12,.1)
    ax2.plot(x,gammadist(x),'k-')
    if index<=10:
        y=np.zeros(len(x))
        ax2.plot(x,y)
    else:
        density=scipy.stats.gaussian_kde(X[:index+1])
        ax2.plot(x,density(x),'r-')

    ax2.set_xlim([0.,12])
    ax2.set_ylim([0.,2])
    ax2.set_xlabel('X')
    ax2.set_ylabel('density')

ani_chain=animation.FuncAnimation(fig,anima_chain,interval=1)
ani_density=animation.FuncAnimation(fig,anima_density,interval=1)

def main():
    plt.show()

if __name__=='__main__':
    main()
"""
#测试

"""
def MCMC(gm, classifier, n_samples, sigma=0.1): #MCMC
    sample_z = []

    z = gm.sample(1)[0]
    for i in range(n_samples):
        uniform_rand = np.random.uniform(size=1)
        z_next = np.random.multivariate_normal(z.squeeze(),sigma*np.eye(2)).reshape(1,-1)

        z_combined = np.concatenate((z, z_next),axis=0)
        scores = cls(torch.Tensor(z_combined)).detach().numpy().squeeze()
        z_score, z_next_score = np.log(scores[0]), np.log(scores[1]) #z score needes to be converted to log, coz gm score is log.
        z_prob, z_next_prob = (gm.score(z)+z_score), (gm.score(z_next)+z_next_score) # two log addition, output: log probability
        accepence = min(0, (z_next_prob - z_prob))

        if i == 0:
            sample_z.append(z.squeeze())

        if np.log(uniform_rand) < accepence:
            sample_z.append(z_next.squeeze())
            z = z_next
        else:
            pass

    return np.stack(sample_z)
"""
"""
latent_poit=torch.load('latent_poit2v.pt').detach().numpy()
mean=np.mean(latent_poit,axis=0)
cov=np.cov(latent_poit,rowvar=False)
#定义高斯分布
gaussian_dist=multivariate_normal(mean,cov)
# 绘制数据点
plt.scatter(latent_poit[:, 0], latent_poit[:, 1], label='Data Points', color='blue', alpha=0.5)

# 创建网格以绘制高斯分布
x = np.linspace(min(latent_poit[:, 0]), max(latent_poit[:, 0]), 100)
y = np.linspace(min(latent_poit[:, 1]), max(latent_poit[:, 1]), 100)
X, Y = np.meshgrid(x, y)

# 计算网格上每个点的高斯分布值
pos = np.dstack((X, Y))
Z = gaussian_dist.pdf(pos)

# 绘制等高线
plt.contour(X, Y, Z, levels=10, cmap='Reds', alpha=0.7)
plt.title('2D Gaussian Fit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

# MCMC参数
num_samples = 1000  # 需要的样本数
sigma_proposal = 0.5  # 提议分布的标准差

# 初始化
x_t = np.random.multivariate_normal(mean, cov)  # 初始点
samples = []

# MCMC采样
for _ in range(num_samples):
    # 采样提议点
    x_star = np.random.multivariate_normal(x_t, sigma_proposal * np.eye(2))

    # 计算接受率
    p_x_star = multivariate_normal.pdf(x_star, mean=mean, cov=cov)
    p_x_t = multivariate_normal.pdf(x_t, mean=mean, cov=cov)

    r = p_x_star / p_x_t

    # 接受或拒绝提议点
    if np.random.rand() < r:
        x_t = x_star

    samples.append(x_t)

# 转换为NumPy数组
samples = np.array(samples)

# 绘制结果
plt.figure(figsize=(10, 10))
plt.scatter(latent_poit[:, 0], latent_poit[:, 1], label='Data Points', color='blue', alpha=0.5)
plt.scatter(samples[:, 0], samples[:, 1], label='MCMC Samples', color='red', alpha=0.3)
plt.title('MCMC Sampling from Fitted 2D Gaussian Distribution')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

"""
"""
def MCMC_sampling(gm, n_samples, proposal_width=1.0):
    """"""
    samples = []
    # 初始化一个点，从 gm（高斯混合模型）中采样
    x = gm.sample(1)[0].squeeze()  # 从 gm 中获取一个初始样本，并展平
    print(gm.sample)


    for i in range(n_samples):
        # 生成提议点，根据正态分布生成
        x_new = np.random.normal(x, proposal_width)


        # 确保 x 和 x_new 的形状为二维
        log_prob_x_new = gm.score_samples(x_new[np.newaxis, :])  # 对 x_new 计算 log 概率
        log_prob_x = gm.score_samples(x[np.newaxis, :])  # 对当前样本 x 计算 log 概率
        # 计算接受率：min(1, exp(log_prob_x_new - log_prob_x))
        acceptance_ratio = min(1,np.exp(log_prob_x_new - log_prob_x))


        # 接受或拒绝提议点
        if np.random.uniform() < acceptance_ratio:
            x = x_new  # 接受新的样本


        samples.append(x)  # 保存当前样本

    return np.array(samples)

# 进行 MCMC 采样
n_samples = 3000
samples = MCMC_sampling(gaussian_dist, n_samples, proposal_width=1.0)
print(len(samples))
# 可视化结果
import matplotlib.pyplot as plt

plt.hist(samples, bins=50, density=True, label='MCMC Samples')
plt.legend()
plt.show()
"""
# 测试
z=gm.sample(1)[0]
print(1)

#引用模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8,32),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

cls=Classifier()
cls.load_state_dict(torch.load('classifer.pth'))
cls.eval()
def MCMC(gm, classifier, n_samples, sigma=0.1): #MCMC
    sample_z = []

    z = gm.sample(1)[0] #样本抽样时返回一个元组，第一个数据是样本数据，第二个数据时标签，表示来自哪个高斯成分
    for i in range(n_samples):
        uniform_rand = np.random.uniform(size=1)

        #从均值为z.sequeeze()，方差为sigma*np.eye(2)二元正太分布中生成一个样本
        #提议矩阵即下面定义的正太分布
        z_next = np.random.multivariate_normal(z.squeeze(),sigma*np.eye(2)).reshape(1,-1)

        z_combined = np.concatenate((z, z_next),axis=0)
        scores = cls(torch.Tensor(z_combined)).detach().numpy().squeeze()
        z_score, z_next_score = np.log(scores[0]), np.log(scores[1]) #z score needes to be converted to log, coz gm score is log.
        #gm.score计算了GMM下的平均对数似然，反应了数据集在模型下的拟合好坏程度,但是这里x只是一个样本点，则gm.score只是对x去对数，十分玄乎
        z_prob, z_next_prob = (gm.score(z)+z_score), (gm.score(z_next)+z_next_score) # two log addition, output: log probability
        #？？？###
        accepence = min(0, (z_next_prob - z_prob))

        if i == 0:
            sample_z.append(z.squeeze())

        if np.log(uniform_rand) < accepence:
            sample_z.append(z_next.squeeze())
            z = z_next
        else:
            pass

    return np.stack(sample_z)

#%%Sample 5000 times with sigma=0.5
sample_z = MCMC(gm=gm, classifier=cls, n_samples=5000, sigma=0.7)
np.save('sample_lanten.npy',sample_z)

print(1)
"""
WAE_comps = model._decode(torch.Tensor(sample_z).to(device)).detach().cpu().numpy()  # new_comps save as csv and goes to TERM
print('Sample size:', sample_z.shape)
WAE_comps=pd.DataFrame(WAE_comps)
WAE_comps.columns=column_name
WAE_comps.to_csv('comps_WAE.csv',index=False)
WAE_comps.head()
"""
