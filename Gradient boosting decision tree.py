import xgboost as xgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import preprocessing
def normalizing_data(data,seed=42):
    #实例化MinMax类
    min_max_scaler=preprocessing.MinMaxScaler()
    #注意dataframe['']返回的是pandas series,实际上是numpy对数组的包装,dataframe[['']]返回的是dataframe
    #a=data[['temperature']],但是经过fit以后还是变回了numpy数组
    df_temperature_scaler=pd.DataFrame(min_max_scaler.fit_transform(data[['temperature']]),columns=['temperature'])
    x=data.drop('strength',axis=1)
    #注意concat需要输入列表
    x_all=pd.concat([data.drop(columns=['temperature','strength'],axis=1), df_temperature_scaler],axis=1)
    y_all=data['strength']
    train_features,test_features,train_labels,test_labels=train_test_split(x_all,y_all,test_size=0.15,random_state=seed)
    return x_all,y_all,train_features,test_features,train_labels,test_labels


# 加载数据集
data = pd.read_csv('data_700.csv')
x_all,y_all,train_features,test_features,train_labels,test_labels=normalizing_data(data)



# 创建DMatrix对象
dtrain = xgb.DMatrix(train_features, label=train_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)

# 设置参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
}

# 训练模型
#model = xgb.train(params, dtrain, num_boost_round=100)

# 训练模型并输出损失
evals = [(dtrain, 'train'), (dtest, 'eval')]
#model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

evals_result = {}  # 用于存储损失值
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals,
                  evals_result=evals_result, early_stopping_rounds=10)

#plt.scatter(evals[0],evals[1])
#plt.show()
plt.figure(figsize=(10, 5))
plt.plot(evals_result['train']['rmse'], label='Train RMSE')
plt.plot(evals_result['eval']['rmse'], label='Eval RMSE')
plt.ylim(0,3)
plt.xlim(60,100)
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.title('Training and Evaluation RMSE')
plt.legend()
plt.grid()
plt.show()

# 预测
y_pred = model.predict(dtest)

# 评估模型
mse = mean_squared_error(test_labels, y_pred)

plt.scatter(test_labels,y_pred)
plt.xlabel('test')
plt.ylabel('pred')
plt.show()
print(f'Mean Squared Error: {mse}')