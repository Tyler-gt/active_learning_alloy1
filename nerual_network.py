import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import preprocessing
#from function_def import normalizing_data
#自定义损失函数
"""
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss,self).__init__()
    def forward(self,output,target):
        loss=torch.mean(torch.abs((target-output)/target))
        return loss
"""

"""
#错误的写法，在写损失函数时应该将,需要在forward中接收参数，不应该在__init__当中
class MAPELoss(nn.Module):
    def __init__(self,output,target):
        super(MAPELoss,self).__init__()
        self.output=output
        self.target=target
    def forward(self):
        loss=torch.mean(torch.abs((self.target-self.output)/self.target))
        return loss
"""


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / targets))
        return loss



#投入模型前最大最小归一化
def minmaxscaler(data):
    min = np.amin(data)
    max= np.amax(data)
    if min==max:
        return data
    return (data-min)/(max-min)

#batch normalization（批归一化），BN2015google，一般用于kaiming初始化之后
#解决梯度消失问题，梯度消失通常由激活函数sigmoid函数引起，由于将数据映射到了0-1之间，
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

#自定义数据集
class FeatureDataset(Dataset):

    #输入x是一个二维的numpy数组
    def __init__(self,x):
        self.x=x
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx])
    def __len__(self):
        return self.x.shape[0]
    def getBatch(self,idxs):
        if idxs==None:
            return idxs
        else:
            x_features=[]
            for i in idxs:
                x_features.append(self.__getitem__(i))
            return torch.FloatTensor(x_features)

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



"""
class Net(nn.Module):
    def __init__(self,n_hidden=216,n_feature=8,n_output=1,w=6):
        super(Net,self).__init__()
        self.Liner1=nn.Linear(n_feature,n_hidden)
        #kaiming初始化
        nn.init.kaiming_normal_(self.Liner1.weight)
        self.hiddens=nn.ModuleList([nn.Linear(n_hidden,n_hidden) for n in range(w)])

        #对模型列表中每一个隐藏层做kaiming初始化
        for m in self.hiddens:
            nn.init.kaiming_normal_(m.weigth)

        #输出层
        self.predict=nn.Linear(n_hidden,n_output)
        nn.init.kaiming_normal_(self.predict.weight)

    def forward(self,x):
        x=self.Liner(x)
        x=F.ReLU(x)

        for m in self.hiddens:
            x=m(x)
            x=F.relu(x)

        x=self.predict(x)
        return x
"""
#另一版
#两版中用到的kaiming初始化可以让模型适合使用relu激活函数
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

def train(net,num_epochs,train_features,test_features,train_labels,test_labels,train_loader,optimizer):
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
    plt.plot(range(1,11),train_ls,label='training loss')
    plt.plot(range(1,11),test_ls,label='test loss')
    plt.show()
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


#数据读取
data=pd.read_csv('data_700.csv')


X,Y,X_train,X_test,Y_train,Y_test=normalizing_data(data)
#实例化
net=Net()
train_loader=DataLoader(TensorDataset(torch.tensor(X_train.to_numpy(),dtype=torch.float32),torch.tensor(Y_train.to_numpy(),dtype=torch.float32)),batch_size=32,shuffle=True)
opt=optim.Adam(net.parameters(),lr=0.001)

#开始训练
#x_train,x_test都是dataframe类型的数据，Y_train,y_test都是series的类型
train(net,num_epochs=1000,train_features=torch.tensor(X_train.values, dtype=torch.float32),train_labels=torch.tensor(Y_train.values, dtype=torch.float32),test_features=torch.tensor(X_test.values, dtype=torch.float32),test_labels=torch.tensor(Y_test.values, dtype=torch.float32),train_loader=train_loader,optimizer=opt)

net.eval()
predict=net(torch.tensor(X.values,dtype=torch.float32))
plt.scatter(predict.detach().numpy(),Y,s=1)
plt.show()