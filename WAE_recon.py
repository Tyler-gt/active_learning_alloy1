import numpy as np
import pandas as pd
import torch

from function_def import normalizing_data,weigths_init
import torch.nn as nn
sample_z=np.load('sample_lanten.npy')

#载入WAE模型准备进行还原
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


WAE_recon=WAE(8)
WAE_recon.load_state_dict(torch.load('Autoencoder_700.pth'))
WAE_recon.eval()
tensor_z=torch.tensor(sample_z,dtype=torch.float32)
alloy_recon=WAE_recon._decoder(tensor_z)
alloy_recon_numpy=alloy_recon.detach().numpy()
df_alloy_recon=pd.DataFrame(alloy_recon_numpy,columns=['Cu','Mg','Mn','Fe','Si','Zn','Zr','temperature'])

#保存还原还原出来的数据
df_alloy_recon.to_csv('alloy_recon.csv')
#WAE_recon=torch.load('Autoencoder_700.pth',WAE)
print(1)