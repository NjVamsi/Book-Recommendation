#!/usr/bin/env python
# coding: utf-8

# - bookID: Unique identification number fro each book
# - title: Name under which book was published
# - authors: Name of the Authors of the book
# - average_rating: Avarage rating of the book recevied in total.
# - isbn: International standarded book number
# - isbn13: 13 digit isbn to identify the book
# - language_code: Primary Language of the book
# - num_pages: Number of pages the book containes
# - ratings_count: Total Number of ratings the book recevied.
# - text_reviews_count: Total number of written reviews recevied.
# - publication_date: Date when the book was first published
# - publisher: Name of the Pulishers

# In[1]:


import pandas as pd
import numpy as np

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# for interactive plots
import ipywidgets
from ipywidgets import interact
from ipywidgets import interact_manual


# In[2]:


df = pd.read_csv("books.csv", error_bad_lines = False)


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.columns = df.columns.str.strip()


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df.describe(include = 'object')


# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated().any()


# In[13]:


df.info()


# # Feature Engineering

# - Extract Important Features
# - Reducing the size of Features
# - Creating new features from the existring ones

# In[14]:


df.columns


# In[15]:


df.isbn.nunique()


# In[16]:


df.isbn13.nunique()


# In[17]:


df.drop(['bookID', 'isbn', 'isbn13'], axis = 1, inplace = True)


# In[18]:


df.columns


# In[19]:


df.publication_date


# In[20]:


df['year'] = df['publication_date'].str.split('/')
df['year'] = df['year'].apply(lambda x: x[2])


# In[21]:


df.head(2)


# In[22]:


df.dtypes


# In[23]:


df['year'] = df['year'].astype('int')


# In[24]:


df.dtypes


# In[25]:


df.columns


# In[26]:


df['year'].min()


# In[27]:


df['year'].max()


# In[28]:


df.columns


# # Exploratory Data Analysis

# In[29]:


df[df['year'] == 2020][['title', 'authors','average_rating','language_code','publisher' ]]


# In[30]:


df.groupby(['year'])['title'].agg('count').sort_values(ascending = False).head(20)


# In[31]:


plt.figure(figsize = (20, 10))
sns.countplot(x = 'authors', data = df,
             order = df['authors'].value_counts().iloc[:10].index)
plt.title("Top 10 Authors with maximum book publish")
plt.xticks(fontsize = 12)
plt.show()


# In[32]:


df.columns


# In[33]:


df.language_code.value_counts()


# In[34]:


df.groupby(['language_code'])[['average_rating', 
                               'ratings_count', 
                               'text_reviews_count']].agg('mean').style.background_gradient(cmap = 'Wistia')


# In[35]:


book = df['title'].value_counts()[:20]
book


# In[36]:


# to find most occuring book in our data
plt.figure(figsize = (20, 6))
book = df['title'].value_counts()[:20]
sns.barplot(x = book.index, y = book,
           palette = 'winter_r')
plt.title("Most occuring Books")
plt.xlabel("Number of Occurance")
plt.ylabel("Books")
plt.xticks(rotation = 75, fontsize = 13)
plt.show()


# In[37]:


sns.distplot(df['average_rating'])
plt.show()


# In[38]:


df[df.average_rating == df.average_rating.max()][['title','authors','language_code','publisher']]


# In[39]:


publisher = df['publisher'].value_counts()[:20]
publisher


# In[40]:


publisher = df['publisher'].value_counts()[:20]
sns.barplot(x = publisher.index, y = publisher, palette = 'winter_r')
plt.title("Publishers")
plt.xlabel("Number of Occurance")
plt.ylabel("Publishers")
plt.xticks(rotation = 75, fontsize = 13)
plt.show()


# ### Recommending Books based on Publishers
# ### Recommending Books based on Authors
# ### Recommending Books based on Language

# In[41]:


df.publisher.value_counts()


# In[42]:


df.columns


# In[43]:


def recomd_books_publisheres(x):
    a = df[df['publisher'] == x][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)


# In[44]:


recomd_books_publisheres('Vintage')


# In[45]:


recomd_books_publisheres('Penguin Books')


# In[46]:


@interact
def recomd_books_publishers(publisher_name = list(df['publisher'].value_counts().index)):
    a = df[df['publisher'] == publisher_name][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)


# In[47]:


df.columns


# # based upon Authors

# In[48]:


@interact
def recomd_books_authors(authors_name = list(df['authors'].value_counts().index)):
    a = df[df['authors'] == authors_name][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)


# In[49]:


df.columns


# In[50]:


@interact
def recomd_books_lang(language = list(df['language_code'].value_counts().index)):
    a = df[df['language_code'] == language][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)


# # Data Preprocessing

# In[51]:


df.head(2)


# In[52]:


def num_to_obj(x):
    if x >0 and x <=1:
        return "between 0 and 1"
    if x > 1 and x <= 2:
        return "between 1 and 2"
    if x > 2 and x <=3:
        return "between 2 and 3"
    if x >3 and x<=4:
        return "between 3 and 4"
    if x >4 and x<=5:
        return "between 4 and 5"
df['rating_obj'] = df['average_rating'].apply(num_to_obj)


# In[53]:


df['rating_obj'].value_counts()


# In[54]:


rating_df = pd.get_dummies(df['rating_obj'])
rating_df.head()


# In[55]:


df.columns


# In[56]:


language_df = pd.get_dummies(df['language_code'])
language_df.head()


# In[57]:


features = pd.concat([rating_df,language_df, df['average_rating'],
                    df['ratings_count'], df['title']], axis = 1)
features.set_index('title', inplace= True)
features.head()


# In[58]:


from sklearn.preprocessing import MinMaxScaler 


# In[59]:


scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)


# In[60]:


features_scaled


# # Model Building

# In[62]:


from sklearn import neighbors


# In[64]:


model = neighbors.NearestNeighbors(n_neighbors=5, algorithm = 'ball_tree',
                                  metric = 'euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)


# In[66]:


df['title'].value_counts()


# In[67]:


@interact
def BookRecomender(book_name = list(df['title'].value_counts().index)):
    book_list_name = []
    book_id = df[df['title'] == book_name.index]
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df.iloc[newid].title)
    return book_list_name


# In[ ]:




