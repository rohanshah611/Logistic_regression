#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls',index_col='MouseID')
df.head()


# To deal with missing values we create a new data frame df1 where the missing values in each column are filled with the mean values from the non-missing values.
# 

# In[3]:


df1=df.fillna(df.mean())
df1.shape


# We will first predict the binary class label in df1['Genotype'] which indicates if the mouse has Down's syndrome or not. String values in df1['Genotype'].values are converted to a numeric vector y with 0 or 1

# In[4]:


temp=df1['Genotype'].values
y=np.unique(temp,return_inverse=True)[1]
print(y)


# Predictions are done using the Standardized expression level of 77 genes

# In[19]:


temp1=df1.values
Xs=temp1[:,:77]
scaler = StandardScaler()
Xs = scaler.fit_transform(Xs)
print(Xs)


# In[20]:


scaler = StandardScaler()
Xs = scaler.fit_transform(Xs)


# In[21]:


logreg = linear_model.LogisticRegression(C=1e5, max_iter=1000)
logreg.fit(Xs, y)


# In[22]:


yhat = logreg.predict(Xs)
acc = np.mean(yhat == y)
print("Accuracy on training data = %f" % acc)


# In[26]:


coef=logreg.coef_
#print(coef)
plt.stem(coef[0,:],use_line_collection='True')
coef1=np.reshape(coef.T,(77,))
coef1=np.sort(coef1)
i=coef1[75:][0]
j=coef1[75:][1]
print('indices of the maximum coefficeint vector= {},{}'.format(i,j))


# These are the genes that are
# likely to be most involved in Down's Syndrome

# The above meaured the accuracy on the training data. It is more accurate to measure the accuracy
# on the test data. Perform 10-fold cross validation and measure the average precision, recall and
# f1-score, as well as the AUC

# In[28]:


from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
nfold = 10
kf = KFold(n_splits=nfold,shuffle=True)
prec = []
rec = []
f1 = []
acc = []
for train, test in kf.split(Xs):
 # Get training and test data
    Xtr = Xs[train,:]
    ytr = y[train]
    Xts = Xs[test,:]
    yts = y[test]

 # Fit a model
    logreg.fit(Xtr, ytr)
    yhat = logreg.predict(Xts)

 # Measure performance
    preci,reci,f1i,_= precision_recall_fscore_support(yts,yhat,average='binary')
    prec.append(preci)
    rec.append(reci)
    f1.append(f1i)
    acci = np.mean(yhat == yts)
    acc.append(acci)
# Take average values of the metrics
precm = np.mean(prec)
recm = np.mean(rec)
f1m = np.mean(f1)
accm= np.mean(acc)
# Compute the standard errors
prec_se = np.std(prec)/np.sqrt(nfold-1)
rec_se = np.std(rec)/np.sqrt(nfold-1)
f1_se = np.std(f1)/np.sqrt(nfold-1)
acc_se = np.std(acc)/np.sqrt(nfold-1)
print('Precision = {0:.4f}, SE={1:.4f}'.format(precm,prec_se))
print('Recall = {0:.4f}, SE={1:.4f}'.format(recm, rec_se))
print('f1 = {0:.4f}, SE={1:.4f}'.format(f1m, f1_se))
print('Accuracy = {0:.4f}, SE={1:.4f}'.format(accm, acc_se))


# In[ ]:





# Multi-Class Classification:
# Demonstration of multi-class calssification on df['class'].
# This has 8 possible classes.

# In[31]:


temp=df1['class'].values
y=np.unique(temp,return_inverse=True)[1]
print(y)


# In[32]:


logreg = linear_model.LogisticRegression(C=1,multi_class='ovr')
logreg.fit(Xs, y)


# In[33]:


yhat = logreg.predict(Xs)
acc = np.mean(yhat == y)
print("Accuracy on training data = %f" % acc)


# In[38]:


nfold = 10
kf = KFold(n_splits=nfold,shuffle=True)
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
for train, test in kf.split(Xs):
    prec = []
    rec = []
    f1 = []
    acc = []
 # Get training and test data
    Xtr = Xs[train,:]
    ytr = y[train]
    Xts = Xs[test,:]
    yts = y[test]
    
 # Fit a model
    logreg.fit(Xtr, ytr)
    yhat = logreg.predict(Xts)
    
    c1=confusion_matrix(yts, yhat, labels=None, sample_weight=None)
    c=preprocessing.normalize(c1)
    print(np.array_str(c, precision=4, suppress_small=True)) 
    preci,reci,f1i,_= precision_recall_fscore_support(yts,yhat)
    prec.append(preci)
    rec.append(reci)
    f1.append(f1i)
    acci = np.mean(yhat == yts)
    acc.append(acci)
precm = np.mean(prec)
recm = np.mean(rec)
f1m = np.mean(f1)
accm= np.mean(acc)
# Compute the standard errors
prec_se = np.std(prec)/np.sqrt(nfold-1)
rec_se = np.std(rec)/np.sqrt(nfold-1)
f1_se = np.std(f1)/np.sqrt(nfold-1)
acc_se = np.std(acc)/np.sqrt(nfold-1)
print('Precision = {0:.4f}, SE={1:.4f}'.format(precm,prec_se))
print('Recall = {0:.4f}, SE={1:.4f}'.format(recm, rec_se))
print('f1 = {0:.4f}, SE={1:.4f}'.format(f1m, f1_se))
print('Accuracy = {0:.4f}, SE={1:.4f}'.format(accm, acc_se))


# In[39]:


ytr = logreg.predict(Xtr)
coef=logreg.coef_
coef.shape
plt.stem(coef[0,:])


# Line plot of coefficients

# In[ ]:




