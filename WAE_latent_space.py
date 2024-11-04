import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from function_def import min_max_normal

#初始化定义（weigth_init）
def weigths_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias,0)


# 定义模型

"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 压缩为2维
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

"""

"""
class WeightedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, weights=None):
        super(WeightedAutoencoder, self).__init__()

        #初始化权重，如果没有提供则权重默认为1
        if weights is not None:
            self.weights=nn.Parameter(weights)    #可学习权重
        else:
            self.weights=nn.Parameter(torch.ones(input_dim))

        #self.weights = weights if weights is not None else torch.ones(input_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 压缩为2维
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # 对输入加权
        weighted_x = x * self.weights

        # 编码和解码
        encoded = self.encoder(weighted_x)
        decoded = self.decoder(encoded)
        return decoded

"""


# 定义输入维度与隐藏层维度
input_dim = 8
hidden_dim = 10

# 实例化模型并加载权重
model_Autoencoder = WeightedAutoencoder(input_dim, hidden_dim)
model_Autoencoder.load_state_dict(torch.load("Autoencoder.pth"))

# 加载CSV文件数据
data = pd.read_csv('data_final.csv')
df_competition = data[['Cu', 'Mg', 'Mn', 'Fe', 'Si', 'Zn', 'Zr', 'temperature']]
#从dataframe中随机抽取700组数据
df_competition_sample=df_competition.sample(n=700,replace=True)
#df_competition归一化

# 转换为张量，确保没有NaN值并且数据类型是float32
competition_tensor = torch.tensor(df_competition_sample.dropna().values, dtype=torch.float32)

# 通过编码器获取特征点
latent_point2v = model_Autoencoder.encoder(competition_tensor)
print(latent_point2v)
# 绘制二维散点图
plt.scatter(latent_point2v[:, 0].detach().numpy(), latent_point2v[:, 1].detach().numpy())
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space')
plt.show()
