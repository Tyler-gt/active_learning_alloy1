import pandas as pd

target_NN = pd.read_csv('BO_NN')  # get the best BO parameter for NN
lr = target_NN.at[1, 'lr']

print(1)