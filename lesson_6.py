#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[14]:


from sklearn.datasets import load_boston


# In[15]:


boston=load_boston()


# In[16]:


boston.keys()


# In[17]:


data=boston.data


# In[18]:


data.shape


# In[19]:


data


# In[20]:


target=boston.target


# In[21]:


feature_names=boston.feature_names


# In[22]:


feature_names


# In[23]:


for line in boston.DESCR.split('\n'):
    print(line)


# In[24]:


X=pd.DataFrame(data, columns=feature_names)


# In[25]:


X.head()


# In[26]:


X.shape


# In[31]:


y=pd.DataFrame(target, columns=['price'])


# In[32]:


y.head()


# In[33]:


y.info()


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lr=LinearRegression()


# In[38]:


lr.fit(X_train, y_train)


# In[39]:


y_pred=lr.predict(X_test)


# In[41]:


check_test=pd.DataFrame({'y_test': y_test['price'],
                             'y_pred': y_pred.flatten()},
                            columns=['y_test' , 'y_pred'])


# In[42]:


check_test.head(10)


# In[43]:


check_test['error'] = check_test['y_pred'] - check_test['y_test']


# In[44]:


check_test.head()


# In[45]:


initial_mse = (check_test['error']** 2).mean()
initial_mse


# In[46]:


from sklearn.metrics import mean_squared_error


# In[48]:


initial_mse = mean_squared_error(y_test, y_pred)
initial_mse


# In[49]:


(np.abs(check_test['error'])).mean()


# In[50]:


from sklearn.metrics import mean_absolute_error


# In[51]:


mean_absolute_error(y_test, y_pred)


# In[52]:


from sklearn.metrics import r2_score


# In[53]:


r2_score(y_test, y_pred)


# In[54]:


from sklearn.ensemble import RandomForestRegressor


# In[55]:


parameters = [{'n_estimators': [1000],
              'max_depth': [12],
              'random_state': [42]}]


# In[56]:


fr=RandomForestRegressor()


# In[57]:


fr.fit(X_train, y_train.values[:, 0])


# In[58]:


y_pred=fr.predict(X_test)


# In[59]:


check_test=pd.DataFrame({'y_test': y_test['price'],
                             'y_pred': y_pred.flatten()},
                            columns=['y_test' , 'y_pred'])


# In[60]:


check_test.head(10)


# In[62]:


check_test['error'] = check_test['y_pred'] - check_test['y_test']


# In[63]:


check_test.head()


# In[65]:


initial_mse = mean_squared_error(y_test, y_pred)
initial_mse


# In[66]:


mean_absolute_error(y_test, y_pred)


# In[67]:


r2_score(y_test, y_pred)


# RamdomForestRegressor model is better than previous LinearRegression

# In[ ]:




