import torch
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import KFold
from torch.optim import Adam
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

# 读入数据
df_700 = pd.read_csv('data_700.csv')
a = np.mean(df_700['strength'])
y_tensor = torch.tensor((df_700['strength'] > 370).astype(int))
x_tensor = torch.load('latent_poit2v_700.pt')


# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义分类器模型

class Classifier(nn.Module):
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,8),
            nn.Dropout(0.5),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    """

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



"""
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc=nn.Linear(2,8)
        self.fc2=nn.Linear(8,32)
        self.fc3=nn.Linear(32,8)
        self.fc4=nn.Linear(8,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x1=self.fc(x)
        x1=self.relu(x1)
        x2=self.fc2(x1)
        x2=self.relu(x2)
        x3=self.fc3(x2)
        x3=self.relu(x3)
        x3=x3+x1
        x4=self.fc4(x3)
        x4=self.relu(x4)

        out=self.sigmoid(x4)

        return out
"""
# 实例化模型和优化器
cls = Classifier()
opt = Adam(cls.parameters(), lr=1e-4, weight_decay=0.)

kf = KFold(n_splits=5)


def trainning_Cls(model, optimizer):
    train_acc = []
    test_acc = []
    cls_epoch =150
    k = 1

    for train_idx, test_idx in kf.split(x_tensor):
        # 使用 train 和 test 的索引进行操作
        X_train, X_test, y_train, y_test = x_tensor[train_idx], x_tensor[test_idx], y_tensor[train_idx], y_tensor[
            test_idx]
        train_dataloader = DataLoader(CustomDataset(X_train, y_train), batch_size=8, shuffle=True)
        test_dataloader = DataLoader(CustomDataset(X_test, y_test), batch_size=16, shuffle=False)

        for epoch in range( cls_epoch):
            model.train()
            total_loss = []
            total_acc = []

            for x, y in train_dataloader:
                y_pre = model(x)
                loss = F.binary_cross_entropy(y_pre, y.float().unsqueeze(1))
                total_acc.append((torch.sum((y_pre >= 0.5).int() == y.int().unsqueeze(1)).detach().numpy()))
                #print(total_acc)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            #print(total_acc)

        # 测试阶段
        model.eval()
        with torch.no_grad():
            test_total_acc = []
            for x, y in test_dataloader:
                y_pre = model(x)
                accuracy = torch.sum((y_pre >= 0.5).int() == y.int().unsqueeze(1)).detach().numpy() / y_pre.size(0)
                test_total_acc.append(accuracy)

        # 记录每折的准确率
        print(
            f'[{k}/5] train_acc: {sum(total_acc) / len(train_dataloader.dataset):.04f} || test_acc: {np.mean(test_total_acc):.04f}')
        train_acc.append(sum(total_acc) /len(train_dataloader.dataset))
        print(len(train_dataloader.dataset))
        print(sum(total_acc))
        test_acc.append(np.mean(test_total_acc))
        k += 1

    # 最终输出平均准确率
    print(f'train_acc: {np.mean(train_acc):.04f} || test_acc: {np.mean(test_acc):.04f}')

    # 绘图
    plt.figure()
    sns.set_style("darkgrid")
    plt.xlabel('number of folds')
    plt.ylabel('accuracy')
    x = range(1, 6)
    plt.plot(x, train_acc, label="train_accuracy")
    plt.plot(x, test_acc, linestyle=':', c='steelblue', label="test_accuracy")
    plt.legend()
    plt.show()
    #plt.savefig('figure/binary_classifier.png', dpi=300)

    return train_acc, test_acc


# 训练模型
train_acc, test_acc = trainning_Cls(cls, opt)

torch.save(cls.state_dict(), 'classifer.pth')