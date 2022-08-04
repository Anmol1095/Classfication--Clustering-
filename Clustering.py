#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#import k-means
from sklearn.cluster import KMeans


# In[3]:


ad_data = pd.read_csv('wholesale_customers.csv')
ad_data = ad_data.drop(['Channel','Region'],axis=1)


# In[4]:


ad_data.describe().loc[['mean','min','max'],:]


# ### Initialize algo kmeans

# In[5]:


kmeans = KMeans(n_clusters=3)


# In[6]:


# fit k-means object to the dataset 
kmeans.fit(ad_data)


# In[7]:


len(kmeans.labels_)


# In[ ]:





# In[8]:


names = ["Fresh", "Frozen","Milk","Grocery",
         "Detergents_Paper","Delicassen"]
for x in names:
    for y in names:
        if x!= y:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(x=x,
                        y=y,
                        data=ad_data,
                        c=kmeans.labels_,
                        cmap='rainbow'
                       )

            ax.set(title = "Scatter Plot (KMeans=3)",
                   xlabel = x,
                   ylabel = y)
            plt.show()


# In[ ]:





# In[96]:


# #visulasie the data
# plt.scatter(ad_data[0][:,0],ad_data[0][:,1],c=ad_data[1], cmap='rainbow')


# In[38]:


kmeans.cluster_centers_


# In[39]:


kmeans.labels_


# In[ ]:





# In[ ]:




