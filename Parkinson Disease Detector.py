#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xgboost


# In[3]:


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


#Read the data
df=pd.read_csv('C:/Users/Aditya2406/Downloads/parkinsons.data')
df.head()


# In[5]:


#Get the features and labels
#Features are columns except status
#labels are those in the 'status' column.
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[6]:


#Get the count of each label (0 and 1) in labels
#The 'status' column has values 0 and 1 as labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[7]:


#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[8]:


#Split the dataset into training and testing sets keeping 20% of the data for testing
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[9]:


#Train the model
model=XGBClassifier()
model.fit(x_train,y_train)


# In[10]:


#Calculate the accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




