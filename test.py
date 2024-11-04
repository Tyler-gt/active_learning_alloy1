import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
"""
# 示例数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 初始化归一化器
scaler = MinMaxScaler()

# 归一化输入数据
data_scaled = scaler.fit_transform(data)

# 转换为 PyTorch 的张量
data_scaled = torch.tensor(data_scaled, dtype=torch.float32)


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 初始化模型、损失函数和优化器
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    # 前向传播
    output = autoencoder(data_scaled)
    loss = criterion(output, data_scaled)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 预测并反归一化输出
with torch.no_grad():
    data_decoded = autoencoder(data_scaled).numpy()

# 反归一化
data_original = scaler.inverse_transform(data_decoded)

print("原始数据：", data)
print("解码后数据（反归一化后）：", data_original)
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. 定义均值和协方差矩阵
mean = [0, 0]  # 均值
cov = [[1, 0.5], [0.5, 1]]  # 协方差矩阵

# 2. 创建网格点 (x, y) 用于绘图
x, y = np.mgrid[-3:3:.01, -3:3:.01]  # 创建一个从 -3 到 3 的网格，步长为 0.01
pos = np.dstack((x, y))  # 将 x 和 y 网格点组合成坐标对

# 3. 创建一个二维高斯分布
rv = multivariate_normal(mean, cov)

# 4. 绘制二维高斯分布的等高线图
plt.figure(figsize=(6, 6))
plt.contourf(x, y, rv.pdf(pos), cmap='viridis')  # 绘制填充等高线图
plt.colorbar()  # 添加颜色条
plt.title('2D Gaussian Distribution Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

# 定义高斯分布的参数
mu = 0
sigma = 1
n_samples = 10000


# 定义目标分布（高斯分布的概率密度函数）
def target_distribution(x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Metropolis-Hastings采样
def metropolis_hastings(start, n_samples):
    samples = [start]
    current = start

    for _ in range(n_samples - 1):
        proposal = np.random.normal(current, 0.5)  # 提议分布
        acceptance_ratio = target_distribution(proposal) / target_distribution(current)

        if np.random.rand() < acceptance_ratio:
            current = proposal

        samples.append(current)

    return np.array(samples)


# 开始采样
initial_value = 0
samples = metropolis_hastings(initial_value, n_samples)
print(len(samples))
# 可视化采样结果
plt.hist(samples, bins=30, density=True, alpha=0.5, label='MCMC Samples')
x = np.linspace(-4, 4, 100)
plt.plot(x, target_distribution(x), label='Target Distribution', color='red')
plt.legend()
plt.show()

"""import torch
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import KFold
from torch.optim import Adam, lr_scheduler
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
#读入数据,
df_80=pd.read_csv('data_80.csv')
a=np.mean(df_80['strength'])
y_tensor=torch.tensor((df_80['strength'] > 370).astype(int))
x_tensor=torch.load('latent_poit2v.pt')

#作图分析强度分布，用作分类划分0，1的区别
#plt.hist(df_80['strength'])
#plt.show()
#b=y_tensor[[1,2,3,4]] 测试tensor是否可以直接列表调用，为下面kf提供依据

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(2,8),
            nn.Dropout(0.5),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.fc(x)

cls=Classifier()
opt=Adam(cls.parameters(),lr=1e-4,weight_decay=0.)
kf=KFold(n_splits=5)
#b,c=kf.split(x_tensor)
print(1)


def trainning_Cls(model,optimizer):
    train_acc = []
    test_acc = []

    cls_epoch=30
    k=1
    for train, test in kf.split(x_tensor):
        # 返回的train,test均为一个列表，存储的是数据的索引
        # 使用 train 和 test 的索引进行操作
        # print(train)
        # print(test)
        X_train, X_test, y_train ,y_test= x_tensor[train], x_tensor[test], y_tensor[train],y_tensor[test]
        train_dataloader=DataLoader(Dataset(X_train,y_train),batch_size=8,shuffle=True)
        test_dataloader=DataLoader(Dataset(X_test,y_test),batch_size=16,shuffle=False)

        for epoch in range(cls_epoch):
            model.train()
            total_loss=[]
            total_acc=[]
            for i,data in enumerate(train_dataloader):
                x=data[0]
                y=data[1]
                y_pre=model(x)
                loss=F.binary_cross_entropy(y_pre,y)
                #(y_pre>=0.5)会返回布尔张良[True,False,True,False],然后通过.int()强制转化为整形张量
                total_acc.append(torch.sum((y_pre >= 0.5).int() == y.int()).detach().numpy())
                total_loss.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            for i,data in enumerate(test_dataloader):
                x=data[0]
                y=data[1]
                y_pre=model(x)
                loss=F.binary_cross_entropy(y_pre,y)
                #(y_pre>=0.5)会返回布尔张良[True,False,True,False],然后通过.int()强制转化为整形张量
                accuracy=torch.sum((y_pre >= 0.5).int() == y.int()).detach().numpy()/y_pre.size(0)
                test_loss=F.binary_cross_entropy(y_pre,y)

        print('[{}/{}] train_acc: {:.04f} || test_acc: {:.04f}'.format(k, 5,sum(total_acc) / 80,accuracy.item()))
        train_acc.append(sum(total_acc) / 80)
        test_acc.append(accuracy.item())
        k += 1

    print('train_acc: {:.04f} || test_acc: {:.04f}'.format(sum(train_acc) / len(train_acc),
                                                           sum(test_acc) / len(test_acc)))
    plt.figure()
    sns.set_style()
    plt.xlabel('number of folds')
    plt.ylabel('loss')
    x = range(1, 5 + 1)
    sns.set_style("darkgrid")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    plt.plot(x, train_acc)
    plt.plot(x, test_acc, linestyle=':', c='steelblue')
    plt.legend(["train_accuracy", "test_accuracy"])
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('figure/binary_classifier.png', dpi=300)
    return train_acc, test_acc


train_acc, test_acc = trainning_Cls(cls, opt)



"""
