import datetime
import json

import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import time
import os
import pickle
import seaborn as sns
from tqdm import tqdm
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / targets))
        return loss

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
    train_features,test_features,train_labels,test_labels=train_test_split(x_all,y_all,test_size=0.0005,random_state=seed)
    return x_all,y_all,train_features,test_features,train_labels,test_labels



data=pd.read_csv('data_700.csv')
x_all,y_all,train_features,test_features,train_labels,test_labels=normalizing_data(data)
train_features = torch.tensor(train_features.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels.values, dtype=torch.float32)
#test_features = torch.tensor(test_features.values, dtype=torch.float32)
#test_labels = torch.tensor(test_labels.values, dtype=torch.float32)


def ensemble_model(train_features, test_features, train_labels, test_labels, i):    #test_labels赋予什么值并无意义，并没有有用到只是测试方便
    # NN
    target_NN = pd.read_csv('BO_NN')  # get the best BO parameter for NN
    lr = target_NN.at[i, 'lr']  # i means the ith parameter in the BO results
    module_n_hidden = target_NN.at[i, 'module_n_hidden']
    module_w = target_NN.at[i, 'module_w']
    batch_size = target_NN.at[i, 'batch_size']

    module_n_hidden = int(module_n_hidden)
    module_w = int(module_w)
    batch_size = int(batch_size)

    # print('batch_size is {}'.format(batch_size))

    train_dataset = Data.TensorDataset(train_features, train_labels)
    # test_dataset = Data.TensorDataset(test_features, test_labels)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    # test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True)

    module_n_hidden = int(module_n_hidden)
    module_w = int(module_w)
    batch_size = int(batch_size)

    # training NN
    net = Net(n_feature=8, n_hidden=module_n_hidden, n_output=1, w=module_w)

    train_ls, test_ls = [], []
    loss = MAPELoss()
    n_epochs = 1000
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    # print('n_epochs is {}'.format(n_epochs))
    for epoch in range(n_epochs):
        for x, y in train_loader:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())

        #测试
        if epoch % 100 == 0:
            print ("epoch %d: train loss %f" % (epoch, train_ls[-1]))
    # predicting
    net.eval()
    NN_predict = net(torch.tensor(test_features.values,dtype=torch.float32)).detach().numpy()

    # GBDT
    target_GBDT = pd.read_csv('BO_GBDT')  # get the best BO parameter for GBDT
    colsample_bytree = target_GBDT.at[i, 'colsample_bytree']
    learning_rate = target_GBDT.at[i, 'learning_rate']
    max_bin = target_GBDT.at[i, 'max_bin']
    max_depth = target_GBDT.at[i, 'max_depth']
    max_bin = target_GBDT.at[i, 'max_bin']
    min_child_samples = target_GBDT.at[i, 'min_child_samples']
    min_child_weight = target_GBDT.at[i, 'min_child_weight']
    min_split_gain = target_GBDT.at[i, 'min_split_gain']
    n_estimators = target_GBDT.at[i, 'n_estimators']
    num_leaves = target_GBDT.at[i, 'num_leaves']
    reg_alpha = target_GBDT.at[i, 'reg_alpha']
    reg_lambda = target_GBDT.at[i, 'reg_lambda']
    subsample = target_GBDT.at[i, 'subsample']
    params = {
        "num_leaves": int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'learning_rate': learning_rate,
        'n_estimators': int(round(n_estimators)),
        'max_bin': int(round(max_bin)),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0),
        'max_depth': int(round(max_depth)),
        'reg_lambda': max(reg_lambda, 0),
        'reg_alpha': max(reg_alpha, 0),
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'objective': 'regression',
        'verbose': -1
    }
    model = LGBMRegressor(metric='mape', **params)
    #中途保存
    print(f'GBDT training start:{i}')

    #不要更改，解决bug用的
    train_features_numpy=train_features
    train_features_numpy=train_features_numpy.detach().numpy()
    train_labels_numpy=train_labels
    train_labels_numpy=train_labels_numpy.detach().numpy()
    test_features_numpy=test_features
    test_features_numpy=test_features_numpy.to_numpy()
    model.fit(train_features_numpy, train_labels_numpy)
    #中途展示
    print('GBDT training end_______')
    GBDT_predict = model.predict(test_features_numpy)


    return NN_predict, GBDT_predict


single_predict = pd.DataFrame()
test_features=pd.read_csv('alloy_recon.csv').iloc[:,1:]

#a=test_features.columns[0]
for j in tqdm(range(10)):
    print(1)
    #print ('prediction_round_{}'.format(j))
    NN_predict, GBDT_predict = ensemble_model(train_features, test_features, train_labels, 0, j)
    #print(NN_predict)
    #print(GBDT_predict)


    #示例
    #name = "Alice"
    #age = 30
    #print("My name is {} and I am {} years old.".format(name, age))

    single_predict['pred_Tree_{}'.format(j)] = GBDT_predict
    single_predict['pred_NN_{}'.format(j)] = NN_predict

single_predict.to_csv('ensemble_predict_first.csv')
"""
#test_features=pd.read_csv('alloy_recon.csv')
Predict=[]
for i in range(10):
    Predict.append(ensemble_model(train_features, test_features, train_labels, 0, i))
with open('ensembel_predict.json', 'w') as f:
    json.dump([list(t) for t in Predict], f)
"""

print(1)
