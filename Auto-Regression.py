#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pandas import DataFrame as df
import scipy
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime


# In[3]:


df = pd.read_csv('/Users/sarahcliff/Desktop/hydrology data/deseasoned data/ds1detrended@2022-01-20.csv')
df.dropna(subset = ['Data'], inplace = True)
date = df['Date']
depth = df['Data']
print(depth)


# In[4]:


def days_since(vec):
    days_since = []
    for i in range(0, len(vec)):
        days_since.append((i+1))
    return days_since

date_ds = days_since(date)


# In[5]:


plot_pacf(depth, lags = 20, alpha = 0.01)
plt.title('DS1 partial auto-correlation')
plt.show()
ordermag = 3


# In[6]:


train = depth
model = AutoReg(train, lags=ordermag)
model_fit = model.fit()
predictions = model_fit.predict(dynamic=False)
print(predictions)
plt.plot(depth, label = 'depth')
plt.plot(predictions, color='red', label = 'predictions')
plt.ylabel('depth')
plt.xlabel('data point #')
plt.title('Autoregression of LST Day')
plt.legend()
plt.show()

for i in range(0,ordermag):
    predictions[i] = depth[i]
    


# In[9]:


#adjusted dataset
adjust_data = []
for i in range(len(predictions)):
    adjust_data.append(depth[i] - predictions[i])
    
plt.plot(adjust_data)
plt.title('DS1 after autoregression')
plt.xlabel('days')
plt.ylabel('depth')


# In[118]:


data = {'Date': date, 'Data': adjust_data}  
df_data = pd.DataFrame(data)
#creating filename
filename='lstdayautoreg@'+ str(datetime.now().strftime("%Y-%m-%d"))+'.csv'
df_data.to_csv(filename)


# In[ ]:




