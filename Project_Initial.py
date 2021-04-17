#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install plotly==4.14.3


# In[3]:


import numpy as np
import random
import xlrd
import seaborn as sns
import pandas as pd
import plotly.offline
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import collections
import datetime
import json
import re



# In[4]:


data = pd.read_csv(r'D:\Data_MIning\MoviesOnStreamingPlatforms_updated.csv')
columns = data.columns
print(columns)


# In[5]:


#Showing the first 10 values in the dataset 
data.head(20)


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.isnull().sum()


# In[9]:


data.nunique()


# In[10]:


#Dropping all the duplicate values in the dataSet 
data.drop_duplicates(subset='Title',
                         keep='first',inplace=True)


# In[52]:


#deleted unneccesary columns, unnamed and type
df = data.drop(columns=['Unnamed: 0', 'Type','ID'])


# In[53]:


#Filling the missing values in IMDB rating and Rotten Tomatoes So that it doesn't affect the analysis
#We are also converting the data type to intger so that analysis could be easy
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna('0%')
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(lambda x : x.rstrip('%'))
df['Rotten Tomatoes'] = pd.to_numeric(df['Rotten Tomatoes'])

df['IMDb'] = df['IMDb'].fillna(0)
df['IMDb'] = df['IMDb']*10
df['IMDb'] = df['IMDb'].astype('int')


# In[54]:


df.head(10)


# In[55]:


columns=[
        'Title',  'Directors',
      'Country', 'Language'
]
 

df['Title'] = df['Title'].apply(lambda x: x.lower())


# In[56]:


df.head(2)


# In[57]:


# The analysis of countries in which the movies were produced
plt.figure(figsize=(30,10))
sns.countplot(x='Country', data=df, order=df.Country.value_counts().index[0:15])


# In[ ]:





# In[59]:


#Number of movies in each streaming plateform
df_long=pd.melt(df[['Title','Netflix','Hulu','Disney+',
                                'Prime Video']],id_vars=['Title'],
                      var_name='StreamingOn', value_name='Present')
df_long = df_long[df_long['Present'] == 1]
df_long.drop(columns=['Present'],inplace=True)
df_combined = df_long.merge(df, on='Title', how='inner')
df_combined.drop(columns = ['Netflix',
                                  'Hulu', 'Prime Video', 'Disney+'], inplace=True)
df_both_ratings = df_combined[(df_combined.IMDb > 0) & df_combined['Rotten Tomatoes'] > 0]
df_combined.groupby('StreamingOn').Title.count().plot(kind='bar')


# In[60]:


# Finding Out which streaming plateform is the best for Subscription according to IMBD and Rotten Tomamtoes Rating
figure = []
figure.append(px.violin(df_both_ratings, x = 'StreamingOn', y = 'IMDb', color='StreamingOn'))
figure.append(px.violin(df_both_ratings, x = 'StreamingOn', y = 'Rotten Tomatoes', color='StreamingOn'))
fig = make_subplots(rows=2, cols=4, shared_yaxes=True)

for i in range(2):
    for j in range(4):
        fig.add_trace(figure[i]['data'][j], row=i+1, col=j+1)

fig.update_layout(autosize=False, width=800, height=800)        
fig.show()


# In[61]:


#Using Scatter Plot to See which Plateform is better for subscription 
px.scatter(df_both_ratings, x='IMDb',
           y='Rotten Tomatoes',color='StreamingOn')


# In[62]:


#The average Runtime of Movie
plt.figure(figsize=(20,15))
plt.title("Movie Length Distribution", fontsize=15)
sns.distplot(df.Runtime)


# In[63]:


plt.figure(figsize=(30,10))
sns.countplot(x='Genres', data=df, order=df.Genres.value_counts().index[0:15])


# In[64]:


plt.figure(figsize=(30,10))
sns.countplot(x='Language', data=df, order=df.Language.value_counts().index[0:15])


# In[ ]:




