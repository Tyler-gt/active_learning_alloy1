import pandas as pd
import numpy as np
import csv

df=pd.read_csv('data_final.csv')
df_sample=df.sample(n=700,random_state=42,axis=0)
#save df_sample_80
df_sample.to_csv('data_700.csv',index=False)



