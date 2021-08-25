#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


boston=load_boston()


# In[6]:


data=boston.data


# In[7]:


feature_names=boston.feature_names


# In[8]:


x=pd.DataFrame(data, columns=feature_names)


# In[9]:


x.head()


# In[10]:


x.shape


# In[12]:


target=boston.target


# In[14]:


y=pd.DataFrame(target, columns=['price'])


# In[15]:


y.head()


# In[16]:


y.info()


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[24]:


scaler=StandardScaler()


# In[25]:


x_train_scaled=scaler.fit_transform(x_train)


# In[26]:


x_test_scaled=scaler.transform(x_test)


# In[27]:


x_train_scaled=pd.DataFrame(x_train_scaled, columns=feature_names)


# In[29]:


x_test_scaled=pd.DataFrame(x_test_scaled, columns=feature_names)


# In[31]:


from sklearn.manifold import TSNE


# In[32]:


tsne=TSNE(n_components=2, learning_rate=250, random_state=42)


# In[33]:


x_train_tsne=tsne.fit_transform(x_train_scaled)


# In[34]:


x_train_tsne


# In[35]:


x_train_tsne.shape


# In[38]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[62]:


plt.scatter(x_train_tsne [:,0], x_train_tsne [:, 1])


# In[63]:


from sklearn.cluster import KMeans


# In[64]:


kmeans=KMeans(init ="random", n_clusters=3, n_init=10, max_iter=100, random_state=42)


# In[65]:


labels_train=kmeans.fit_predict(x_train_scaled)


# In[66]:


pd.value_counts(labels_train)


# In[67]:


labels_test=kmeans.predict(x_test_scaled)


# In[69]:


plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c=labels_train)


# In[70]:


y_train.mean()


# In[114]:


y_train[labels_train==0].mean()


# In[72]:


y_train[labels_train==1].mean()


# In[115]:


y_train[labels_train==2].mean()


# In[112]:


plt.hist(y_train[labels_train==0], bins = 20, density = True, alpha=0.5)
plt.hist(y_train[labels_train==1], bins = 20, density = True, alpha=0.5)
plt.hist(y_train[labels_train==2], bins = 20, density = True, alpha=0.5)
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.xlabel('price')


# In[95]:


x_train.loc[labels_train==0, 'RM'].mean()


# In[96]:


x_train.loc[labels_train==1, 'RM'].mean()


# In[97]:


x_train.loc[labels_train==2, 'RM'].mean()


# In[122]:


plt.hist(x_train.loc[labels_train==0, 'RM'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==1, 'RM'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==2, 'RM'], bins = 20, density = True, alpha=0.5)
plt.xlim(2, 10)
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.xlabel('RM (average number of rooms per dwelling)')


# In[98]:


x_train.loc[labels_train==0, 'CRIM'].mean()


# In[99]:


x_train.loc[labels_train==1, 'CRIM'].mean()


# In[88]:


x_train.loc[labels_train==2, 'CRIM'].mean()


# In[133]:


plt.hist(x_train.loc[labels_train==0, 'CRIM'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==1, 'CRIM'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==2, 'CRIM'], bins = 20, density = True, alpha=0.5)
plt.xlim(0, 1)
plt.ylim(0, 14)
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.xlabel('CRIM (per capita crime rate by town)')


# In[105]:


plt.hist(x_train.loc[labels_train==0, 'NOX'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==1, 'NOX'], bins = 20, density = True, alpha=0.5)
plt.hist(x_train.loc[labels_train==2, 'NOX'], bins = 20, density = True, alpha=0.5)
plt.xlim(0.4, 1)
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.xlabel('NOX\nnnitrix oxides concentration (parts per 10 million)')


# In[117]:


x_train.loc[labels_train==0, 'NOX'].mean()


# In[118]:


x_train.loc[labels_train==1, 'NOX'].mean()


# In[107]:


x_train.loc[labels_train==2, 'NOX'].mean()


# In[ ]:




