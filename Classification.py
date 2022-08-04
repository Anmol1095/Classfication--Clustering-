#!/usr/bin/env python
# coding: utf-8

# In[453]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[454]:


new_data = pd.read_csv('adult.csv')


# In[455]:


new_data = new_data.drop(['fnlwgt'],axis=1)


# In[456]:


new_data.head()


# In[457]:


new_data


# In[458]:


# question 1 : number of instances
new_data.count()


# In[459]:


# question 1 
new_data.shape


# In[566]:


# (iii) fraction of missing values over all attribute values =( 6465/683788)*100
48842*14


# In[569]:


( 6465/683788)*100


# In[461]:


#(ii) question 1: number of missing values, 
new_data.isnull()


# In[462]:


data_num_missing = new_data.sum()
data_num_missing 


# In[463]:


#(ii) question 1
new_data.isnull().sum().sum()


# In[464]:


new_data.isnull().sum()


# In[465]:


len(new_data)


# In[571]:


#number of instances with missing values 
new_data.shape[0] - new_data.dropna().shape[0]


# In[572]:


new_data.apply(lambda x: x.count(), axis=1)


# In[ ]:





# In[ ]:


#question 2


# In[492]:


for col in new_data.columns:
    print(col)
    print('\n')
    print(new_data[col].unique().tolist())
    print('\n')
        


# In[ ]:


# QUESTION 3


# In[471]:


#df[['one', 'two', 'three']] = df[['one', 'two', 'three']].astype(str)

new_data[['age','workclass','education', 'education-num','marital-status',
          'occupation','relationship','race','sex','capitalgain',
          'capitalloss','hoursperweek','native-country']]  =  new_data[['age','workclass','education', 'education-num','marital-status',
                                                                        'occupation','relationship','race','sex','capitalgain',
                                                                        'capitalloss','hoursperweek','native-country']].astype(str)
   


# In[472]:


from sklearn.preprocessing import LabelEncoder


# In[473]:


new_data_transform = new_data.apply(LabelEncoder().fit_transform)


# In[474]:


new_data_transform['class'].value_counts()


# In[324]:


# train_test_split


# In[ ]:





# In[476]:


from sklearn.model_selection import train_test_split        


# In[477]:


X = new_data_transform.drop('class',axis=1)
y = new_data_transform['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:





# In[478]:


from sklearn.tree import DecisionTreeClassifier


# In[479]:


dtree = DecisionTreeClassifier()


# In[480]:


dtree.fit(X_train,y_train)


# In[482]:


predictions = dtree.predict(X_test)


# In[483]:


from sklearn.metrics import classification_report,confusion_matrix


# In[484]:


print(classification_report(y_test,predictions))


# In[485]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:


# QUESTION 4


# In[ ]:


new_data


# In[326]:


null_data = new_data[new_data.isnull().any(axis=1)]


# In[327]:


new_data.isnull()


# In[328]:


is_NaN = new_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = new_data[row_has_NaN]

print(rows_with_NaN)


# In[329]:


new_data.iloc[:].isnull().sum()


# In[330]:


new_data.index[new_data.isnull().any(axis=1)]


# In[331]:


missing_values = ['']


# In[528]:


missing_df


# In[531]:


# new_data = new_data.replace('', np.nan)
new_data = pd.read_csv('adult.csv')
missing_df = new_data[new_data.isnull().any(axis=1)]
full_df = new_data[~new_data.isnull().any(axis=1)]

sample_df = full_df.sample(3620)


# In[532]:


D_hat_new = pd.concat([missing_df, sample_df],axis=0)


# In[533]:


d_hat_1 = D_hat_new.copy()
d_hat_1['workclass_missing' ]= np.where(d_hat_1['workclass'].isnull(), '1', '0')
d_hat_1['occupation_missing'] = np.where(d_hat_1['occupation'].isnull(), '1', '0')
d_hat_1['native-country_missing'] = np.where(d_hat_1['native-country'].isnull(), '1', '0')


# In[536]:


d_hat_2 = D_hat_new.copy()
d_hat_2['workclass'] = np.where(d_hat_2['workclass'].isnull(), 'Private', d_hat_2['workclass'])
d_hat_2['occupation'] = np.where(d_hat_2['occupation'].isnull(), 'Prof-specialty', d_hat_2['occupation'])
d_hat_2['native-country'] = np.where(d_hat_2['native-country'].isnull(), 'United-States', d_hat_2['native-country'])


# In[547]:



d_hat_1[['age','workclass','education', 'education-num','marital-status',
          'occupation','relationship','race','sex','capitalgain',
          'capitalloss','hoursperweek','native-country','workclass_missing','occupation_missing','native-country_missing']]  =  d_hat_1[['age','workclass','education', 'education-num','marital-status',
                                                                                                                                                      'occupation','relationship','race','sex','capitalgain',
                                                                                                                                                      'capitalloss','hoursperweek','native-country','workclass_missing','occupation_missing','native-country_missing']].astype(str)
   


# In[548]:


from sklearn.preprocessing import LabelEncoder


# In[549]:


d_hat_1_transform = d_hat_1.apply(LabelEncoder().fit_transform)


# In[362]:


#train the dataset


# In[551]:


from sklearn.model_selection import train_test_split


# In[552]:


X = d_hat_1_transform.drop('class',axis=1)
y = d_hat_1_transform['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[403]:


#Training a Decision Tree Model for D1 


# In[553]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[554]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[555]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[417]:


#Training a Decision Tree Model for D2


# In[556]:



d_hat_2[['age','workclass','education', 'education-num','marital-status',
          'occupation','relationship','race','sex','capitalgain',
          'capitalloss','hoursperweek','native-country']]  =  d_hat_2[['age','workclass','education', 'education-num','marital-status',
                                                                       'occupation','relationship','race','sex','capitalgain',
                                                                      'capitalloss','hoursperweek','native-country']].astype(str)
   


# In[557]:


d_hat_2_transform = d_hat_2.apply(LabelEncoder().fit_transform)


# In[558]:


d_hat_2_transform


# In[559]:


X = d_hat_2_transform.drop('class',axis=1)
y = d_hat_2_transform['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[560]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[561]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[562]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




