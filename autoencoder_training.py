import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from function_def import normalizing_data,weigths_init

def normalizing_data_WAE(data,seed=42):
    #实例化MinMax类
    min_max_scaler=preprocessing.MinMaxScaler()
    #注意dataframe['']返回的是pandas series,实际上是numpy对数组的包装,dataframe[['']]返回的是dataframe
    #a=data[['temperature']],但是经过fit以后还是变回了numpu数组
    df_temperature_scaler=pd.DataFrame(min_max_scaler.fit_transform(data[['temperature']]),columns=['temperature'])
    x=data.drop('strength',axis=1)
    #注意concat需要输入列表
    x_all=pd.concat([data.drop(columns=['temperature','strength'],axis=1), df_temperature_scaler],axis=1)
    y_all=data['strength']
    return x_all


#测试
df_80=pd.read_csv('data_80.csv')
a=df_80.columns
df_norm=normalizing_data(df_80)


df_700=pd.read_csv('data_700.csv')
df_norm_700=normalizing_data_WAE(df_700)
print(1)
"""
#简单的自编码器
class Autoencoder(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Autoencoder,self).__init__()

        #Encoder
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2), #"//"整数除法
            nn.ReLU()
        )

        #Decoder
        self.decoder=nn.Sequential(
            nn.Linear(2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            #nn.Sigmoid()
        )

    def forward(self,x):
        encoded=self.encoder(x)
        #print(encoded)
        decoded=self.decoder(encoded)

        return decoded


"""
"""
#定于加权自编码器（WAE）
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
        return decoded,encoded
"""
class WAE(nn.Module):
    def __init__(self, input_size):
        super(WAE, self).__init__()
        self.input_size = input_size

        # encoder
        self.encoder = nn.Sequential(
                        nn.Linear(self.input_size, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 2),
                        )

        # decoder
        self.decoder = nn.Sequential(
                        nn.Linear(2, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, self.input_size),
                        #nn.Sigmoid() # 转为1以内
                        )

        # apply the weights_init function to initialize the weights
        self.apply(weigths_init)
    def _encoder(self,x):
        return self.encoder(x)
    def _decoder(self,x):
        return self.decoder(x)
    def forward(self,x):
        z=self._encoder(x)
        x_recon=self._decoder(z)
        return x_recon,z
"""
#合金数据读取
df=pd.read_csv('data_final.csv')

df=df[['Cu','Mg','Mn','Fe','Si','Zn','Zr','temperature']]
#Min-Max归一化处理
# Min-Max 归一化
normalized_df = (df - df.min()) / (df.max() - df.min())

#采样700组数据
df_700 = normalized_df.sample(n=700,random_state=42)
"""


#提取多列特征,未作归一化的数据
#features=df[['C','Si','Mn','P','S','Cr','Ni','Mo','V','Nb','Al','Zr','W','N','Ta','Fe','Co','Cu','Mo','Y','Zr','Ti','Zn','Sn','As','Pb','Sb','Bi','Pt','Au','Hg','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',']]
competition=df_norm_700[['Cu','Mg','Mn','Fe','Si','Zn','Zr','temperature']]

#将dataframe转变为2tensor张量
alloy_tensor=torch.tensor(competition.values,dtype=torch.float32)

#使用dataloader进行数据加载
#Dataloader=DataLoader(alloy_tensor,batch_size=32,shuffle=True)
data_loader=DataLoader(TensorDataset(alloy_tensor,alloy_tensor),batch_size=32,shuffle=True)

#初始化模型，损失函数与优化器
input_dim=alloy_tensor.shape[1]
#hidden_dim=10
#weigth=torch.tensor([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],dtype=torch.float32)
model=WAE(input_dim)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

#模型训练
num_epochs=3000
loss_=[]
best_loss=np.inf
patience=200
patience_counter=0
for epoch in range(num_epochs):
    for input,_ in data_loader:
        outputs,latent=model(input)
        loss=(criterion(outputs,input))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_.append(loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #早停机制并且保存最优模型
    if best_loss>loss:
        best_loss=loss
        torch.save(model.state_dict(), 'Autoencoder_700.pth')
        #重置计数器
        patience_counter=0
    else:
        patience_counter+=1

    if patience_counter>=patience:
        print("提前停止训练")
        print(f"最低损失loss:{best_loss:-4f}")
        break




plt.plot(range(len(loss_)),loss_)
plt.show()

torch.save(model.state_dict(), 'Autoencoder_700.pth')

#测试编码器的浅层空间的输出
latent_point2v=model._encoder(alloy_tensor)
torch.save(latent_point2v,'latent_poit2v_700.pt')
#测试print(b)
#print(model_WeightedAutoencoder.weights)
plt.scatter(latent_point2v[:, 0].detach().numpy(), latent_point2v[:, 1].detach().numpy())
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space')
plt.show()

#输入-自编码器——输出结果保存
x_recon,x_latent=model(alloy_tensor)

column_names=competition.columns

#将auto_data转变为csv文件
auto_data=pd.DataFrame(x_recon.detach().numpy(),columns=column_names)
auto_data.to_csv('autoencoder_data_700.csv',index=False)
#print(1)

#绘制输入输出比对图（查看效果）(怎么做)


#随机采样潜在空间中的点
#model_Autoencoder = torch.load('Autoencoder.pth')
def sample_latent_vector(latent_dim):
    return torch.randn(10,latent_dim)
def generate_data_from_latent_vector(decoder,latent_vector):
    return decoder(latent_vector)


#生成数据
latent_dim=2  #hidden_dim//2
latent_vector = sample_latent_vector(latent_dim)
generated_data = generate_data_from_latent_vector(model._decoder,latent_vector)

print(1)
print(df_norm_700)

df_recon=pd.read_csv('autoencoder_data_700.csv')
df_recon=df_recon.head(50)
df_norm_700=df_norm_700.head(50)
#plt.subplot(2, 1, 1)  # 2 行 1 列，第 1 个子图
plt.plot(range(df_norm_700.shape[0]),df_norm_700['Cu'],color='red',linestyle='-')
plt.plot(range(df_recon.shape[0]),df_recon['Cu'],color='red',linestyle='--')

plt.plot(range(df_norm_700.shape[0]),df_norm_700['Mg'],color='blue',linestyle='-')
plt.plot(range(df_recon.shape[0]),df_recon['Mg'],color='blue',linestyle='--')

plt.plot(range(df_norm_700.shape[0]),df_norm_700['temperature'],color='c',linestyle='-')
plt.plot(range(df_recon.shape[0]),df_recon['temperature'],color='c',linestyle='--')
plt.legend(['Norm Cu', 'recon Cu', 'Norm Mg', 'recon Mg', 'Norm tem', 'recon tem'])  # 添加图例

plt.show()