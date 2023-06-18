#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dt = pd.read_csv("Bengaluru_House_Data.csv")


# In[3]:


dt.sample(20)


# In[4]:


dt.shape


# In[5]:


dt.info()


# In[6]:


dt.isna().sum()


# In[7]:


dt.drop(columns=['area_type','availability','society','balcony'],inplace=True)


# In[8]:


dt.describe()


# In[9]:


dt.info()


# In[10]:


dt['location'].value_counts()


# In[11]:


dt['location']=dt['location'].fillna('Sarjapur Road')


# In[12]:


dt['size'].value_counts()


# In[13]:


dt['size']=dt['size'].fillna('2 BHK')


# In[14]:


dt['total_sqft'].value_counts()


# In[15]:


dt['total_sqft']=dt['total_sqft'].fillna('1200')


# In[16]:


dt['bath']=dt['bath'].fillna(dt['bath'].median())


# In[17]:


dt.info()


# In[18]:


dt['BHK']=dt['size'].str.split().str.get(0).astype(int)


# In[19]:


dt.head()


# In[20]:


dt.drop(columns=['size'])


# In[21]:


dt['total_sqft'].unique()


# In[22]:


def convertrange(x):
    
    temp= x.split('-')
    if len(temp)== 2:
        return (float(temp[0])+ float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[23]:


dt['total_sqft']=dt['total_sqft'].apply(convertrange)


# In[24]:


dt.describe()


# In[25]:


dt['price_per_sqft']= dt['price']*100000/dt['total_sqft']


# In[26]:


dt['location'].value_counts()


# In[27]:


dt['location']=dt['location'].apply(lambda x : x.strip())
location_count=dt['location'].value_counts()
location_count


# In[28]:


location_count_less10 =location_count[location_count<=10]


# In[29]:


location_count_less10


# In[30]:


dt['location']=dt['location'].apply(lambda x: 'other' if x in location_count_less10 else x)


# In[31]:


dt['location'].value_counts()


# In[32]:


dt.describe()


# In[33]:


(dt['total_sqft']/dt['BHK']).describe()


# In[34]:


dt = dt[((dt['total_sqft']/dt['BHK'])>=300)]


# In[35]:


dt.describe()


# In[36]:


def rem_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft >(m-st))& (subdf.price_per_sqft <=(m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index = True)
    return df_output
dt = rem_outliers_sqft(dt)

dt.describe()                        


# In[37]:


def BHK_outlier_remover(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        BHK_stats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            BHK_stats[BHK]= {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats = BHK_stats.get(BHK-1)
            if stats and stats ['count']> 5:
                exclude_indices =np.append (exclude_indices, BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


# In[38]:


dt=BHK_outlier_remover(dt)


# In[39]:


dt.shape


# In[40]:


dt


# In[41]:


dt.drop(columns=['size','price_per_sqft'],inplace=True)


# # clean data

# In[42]:


dt.head()


# In[43]:


dt.to_csv("Cleaned_data.csv")


# In[44]:


X = dt.drop(columns=['price'])
y = dt['price']


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[46]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 0)


# In[47]:


print(X_train.shape)
print(X_test.shape)


# # Applying Linear Regression
# 

# In[48]:


column_trans = make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')


# In[49]:


scaler = StandardScaler()


# In[50]:


lr = LinearRegression(normalize = True)


# In[51]:


pipe = make_pipeline(column_trans,scaler,lr)


# In[52]:


pipe.fit(X_train,y_train)


# In[53]:


y_pred_lr = pipe.predict(X_test)


# In[55]:


r2_score(y_test, y_pred_lr)


# In[56]:


#Applying Lasso


# In[57]:


lasso = Lasso()


# In[58]:


pipe = make_pipeline(column_trans,scaler, lasso)


# In[59]:


pipe.fit(X_train,y_train)


# In[60]:


y_pred_lasso = pipe.predict(X_test)
r2_score(y_test , y_pred_lasso)


# In[61]:


#apply ridge


# In[62]:


ridge = Ridge()


# In[63]:


pipe = make_pipeline(column_trans,scaler,ridge)


# In[64]:


pipe.fit(X_train,y_train)


# In[66]:


y_pred_ridge = pipe.predict(X_test)
r2_score(y_test ,y_pred_ridge)


# In[68]:


print("No Regularization:" , r2_score(y_test,y_pred_lr))
print("Lasso:" ,r2_score(y_test,y_pred_lasso))
print("Ridge:" ,r2_score(y_test,y_pred_ridge))


# In[69]:


import pickle


# In[70]:


pickle.dump(pipe,open('RidgeModel.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




