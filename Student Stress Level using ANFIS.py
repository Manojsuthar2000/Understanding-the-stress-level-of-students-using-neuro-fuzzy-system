#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd

dataset=pd.read_csv("D:\\sem 5\\INT246\\Dataset\\dataframe_hrv.csv")
print(dataset.shape)

# As you can see the no. of row is very high. So, i will take only rows in which 
# the stress value is distinct.
unique_dataset=dataset.drop_duplicates(subset=['stress'])
print(unique_dataset.shape)

# here, i am trying to check no. of unique values in each rows
print(unique_dataset[:].nunique())


# In[39]:


# i am droping the rows whoes unique count is very less and which are not useful to me
unique_dataset=unique_dataset.drop(['AVNN','RMSSD','TP','ULF','time'], axis = 1)
unique_dataset=unique_dataset.drop(['NNRR','pNN50','VLF','LF','HF','LF_HF'], axis = 1)
unique_dataset=unique_dataset.drop(['Seconds','interval in seconds','marker','newtime','SDNN'], axis = 1)
unique_dataset.shape


# In[40]:


unique_dataset.head()


# In[42]:


# convert few readings from float to int to make ANFIS fast
unique_dataset['HR']=unique_dataset['HR'].astype(int)
unique_dataset['RESP']=unique_dataset['RESP'].astype(int)
unique_dataset['footGSR']=unique_dataset['footGSR'].astype(int)
unique_dataset['handGSR']=unique_dataset['handGSR'].fillna(0).astype(int)
unique_dataset.head()


# In[50]:


min_value=list(unique_dataset.min().values)
min_value # getting min value to calculate mf


# In[53]:


max_value=list(unique_dataset.max().values) 
max_value


# In[46]:


diff=list((unique_dataset.max().values-unique_dataset.min().values)/3) # to find diff for guassmf
diff


# In[57]:


# here i am only taking (HR RESP handGSR) variable for anfis because ANFIS is slow
# making 3 membership function for each variables
import numpy as np

mf=[]
column=unique_dataset.columns
loop=[2,3,5] #(HR RESP handGSR)
for i in loop:
    temp=[]
    low=min_value[i]
    high=min_value[i]+diff[i]
    for j in range(3):
        temp.append(['gaussmf',{'mean':(low+high)/2,'sigma':diff[i]/3}])
        low+=diff[i]
        high+=diff[i]
    mf.append(temp)

for i in range(3):
    print("Membership function for ",column[loop[i]])
    for j in range(3):
        print(mf[i][j])
    print()


# In[58]:


import anfis
from anfis.membership import mfDerivs
from anfis.membership import membershipfunction
from anfis.anfis import ANFIS

mfc = membershipfunction.MemFuncs(mf) # making mfc

X=unique_dataset[['HR','RESP','handGSR']] # training dataset 
X=X.iloc[:200,:] # here, no of column should be atleast 50 for correct training
Y=unique_dataset.iloc[:200,6] # stress column

anf = ANFIS(X, Y, mfc)

# plotting all membership functions

for i in range(len(loop)):
    print("Plot of Membership function for ",column[loop[i]])
    anf.plotMF(np.arange(min_value[loop[i]],max_value[loop[i]]),i)


# In[59]:


anf.trainHybridJangOffLine(epochs=30)

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()


# In[31]:


# by run i understood that only when no. of row is equal to 40 my error is increasing
# else its fine So, if i take no. of row above 50, my error will always be decreasing
for i in range(10,320,10):
    X=unique_dataset[['HR','RESP','handGSR']]
    X=X.iloc[:i,:]
    Y=unique_dataset.iloc[:i,10]
    anf = ANFIS(X, Y, mfc)
    anf.trainHybridJangOffLine(epochs=3)

    print("Plotting errors")
    anf.plotErrors()
    print("Plotting results")
    anf.plotResults()

