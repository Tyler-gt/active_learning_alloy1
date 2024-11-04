import os
import time

import torch
from bayes_opt import BayesianOptimization
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from torch import optim

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

#from nerual_network import MAPELoss


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / targets))
        return loss


def train(net,num_epochs,batch_size,train_features,test_features,train_labels,test_labels,train_loader,optimizer):
    print("\n===train begin ===")
    print(net)
    train_ls,test_ls=[],[]
    loss = MAPELoss()
    for epoch in range(num_epochs):
        for x, y in train_loader:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        if epoch % 100 == 0:
            train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
            test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
            print("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
    #plt.plot(range(1,11),train_ls,label='training loss')
    #plt.plot(range(1,11),test_ls,label='test loss')
    #plt.show()
    print("=== train end ===")

def test(model,test_loader):
    model.eval()
    test_loss=0
    n=0
    loss=MAPELoss()
    with torch.no_grad():
        for data ,target in test_loader:
            output=model(data)
            test_loss+=loss(output.view(-1,1),target.view(-1,1)).item()
            n+=1
    test_loss/=n
    print('Test set: Average loss: {:.4f}'.format(
        test_loss))

    return test_loss
class Net(nn.Module):
    def __init__(self,n_hidden=128,n_feature=8,n_output=1,w=3):
        super(Net,self).__init__()
        self.inputnet=nn.Sequential(
            nn.Linear(n_feature,n_hidden),
            #写法错误nn.init.kaiming_normal(nn.Linear.weigth),
            nn.ReLU()
        )
        """
        写法错误
        self.hiddenet=self.Sequential(
            for m in range(w):
                nn.Linear(n_hidden,n_hidden),
                nn.init.kaiming_normal(nn.Linear.weight),
                nn.ReLU()
        )
        """
        #nn.Sequential是一个类，使用时需要实例化并传入层的列表
        self.hiddens=nn.ModuleList([nn.Linear(n_hidden,n_hidden) for _ in range(w)])
        for m in self.hiddens:
            nn.init.kaiming_normal_(m.weight)

        self.outputnet=nn.Sequential(
            nn.Linear(n_hidden,n_output)

        )
        nn.init.kaiming_normal_(self.outputnet[0].weight)

    def forward(self,x):
        x=self.inputnet(x)
        for m in self.hiddens:
            x=m(x)
            x=F.relu(x)
        x=self.outputnet(x)

        return x
def normalizing_data(data,seed=42):
    #实例化MinMax类
    min_max_scaler=preprocessing.MinMaxScaler()
    #注意dataframe['']返回的是pandas series,实际上是numpy对数组的包装,dataframe[['']]返回的是dataframe
    #a=data[['temperature']],但是经过fit以后还是变回了numpu数组
    df_temperature_scaler=pd.DataFrame(min_max_scaler.fit_transform(data[['temperature']]),columns=['temperature'])
    x=data.drop('strength',axis=1)
    #注意concat需要输入列表
    x_all=pd.concat([data.drop(columns=['temperature','strength'],axis=1), df_temperature_scaler],axis=1)
    y_all=data['strength']
    train_features,test_features,train_labels,test_labels=train_test_split(x_all,y_all,test_size=0.15,random_state=seed)
    return x_all,y_all,train_features,test_features,train_labels,test_labels


data=pd.read_csv('data_700.csv')
x_all,y_all,train_features,test_features,train_labels,test_labels=normalizing_data(data)
train_features = torch.tensor(train_features.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels.values, dtype=torch.float32)
test_features = torch.tensor(test_features.values, dtype=torch.float32)
test_labels = torch.tensor(test_labels.values, dtype=torch.float32)

#下面定义的train_model参数就是需要优化的参数
def train_model(batch_size,lr,module_n_hidden,module_w):
    module_n_hidden=int(module_n_hidden)  #隐藏层神经元的数量
    module_w=int(module_w)
    batch_size=int(batch_size)

    # 创建数据集
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    #初始化模型
    net = Net(n_feature=8, n_hidden=module_n_hidden, n_output=1, w=module_w)

    n_epochs = 1000
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

    # 训练和测试模型
    train(net, n_epochs, batch_size, train_features, test_features,
          train_labels, test_labels, train_loader, optimizer)
    train_loss = test(net, train_loader)
    test_loss = test(net, test_loader)

    return -test_loss  # 返回测试损失的负值用于优化

bounds = {'lr': (0.0005, 0.001), 'batch_size': (32, 64), 'module_n_hidden': (16, 526),
          'module_w': (2, 10)}  # 超参数的搜索范围

optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=1,
)

optimizer.maximize(init_points=50, n_iter=50)  # 进行贝叶斯优化


###############
#保存结果
table = pd.DataFrame(columns=['target', 'batch_size', 'lr', 'module_n_hidden', 'module_w'])
for res in optimizer.res:
    table = table.append(pd.DataFrame({'target': [res['target']],
                                        'batch_size': [res['params']['batch_size']],
                                        'lr': [res['params']['lr']],
                                        'module_n_hidden': [res['params']['module_n_hidden']],
                                        'module_w': [res['params']['module_w']]}), ignore_index=True)

table = table.sort_values(by=['target'], ascending=False)  # 按目标值排序
model_name = 'Invar_BO_NN'
file_name = '{}.xlsx'.format(model_name)
table.to_csv('BO_NN')  # 保存结果为csv文件
print(table)