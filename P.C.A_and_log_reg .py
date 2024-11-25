#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing required modules:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# # Loading the dataset from the Scikit-learn library.

# In[5]:


# Loading the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# # Data Explorations:
# + Reviewing to understand the data. Data description and summary statistics

# In[6]:


# Data overview:

print(data)


# In[7]:


# Data was understood to be of dictionary-type; getting the dictionary keys.

data.keys()


# In[8]:


# Reviewing the dictionary components:

data.data, data.target


# In[9]:


# Describing the data:

print(data.DESCR)


# In[7]:


# checkind feature names:

data.feature_names


# In[8]:


# converting dictionary to pandas data frame for further review:

df= pd.DataFrame(data['data'], columns = data['feature_names'])
df.head()


# In[9]:


# Checking the data frame shape:

df.shape


# In[10]:


# Assigning X and y variables to data component and target respectively:

X = data.data
y = data.target


# In[11]:


# Differences between numerical values are wide, so we employ scaling with Standard scaler or MinMax scaler from scikit-learn
# and fit the data frame to the scaler.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# # PCA implementation

#  ## 1. Implement PCA

# In[12]:


# Iplementing Principal Component Analysis without dimensionality reduction:

pca = PCA(n_components = 30)
X_pca = pca.fit_transform(X_scaled)
X_pca.shape


# In[13]:


# Viewing data after the first PCA implementation:

X_pca


# In[14]:


# Converting data with first PCA implemented to data frame:

pca_df = pd.DataFrame(data=X_pca)
pca_df.head()


# ## 2. Reduce the dataset into 2 PCA components.

# In[15]:


# PCA implementation with dimensionality reduced to two (2) for ease of 2D visualisation and further analysis:

pca_ = PCA(n_components = 2)
X1_pca = pca_.fit_transform(X_scaled)
X1_pca.shape


# In[16]:


# Viewing data reduced to 2 dimensions of Numpy array via PCA:

X1_pca


# In[17]:


# Converting the Numpy array to data frame - dimension reduced to 2.
pca1_df = pd.DataFrame(data=X1_pca)
pca1_df.head()


# In[18]:


# Target variable is y:

pca1_df['target'] = y
y


# In[19]:


# Ploting the PCA components with dimension reduced to 2:

plt.figure(figsize=(8,6))
plt.scatter(pca1_df[0], pca1_df[1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(label='Target')
plt.show()


# # Logistic regression for prediction (optional)

# In[20]:


# Importing the required modules for logistic regression modelling.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[21]:


# Split the data into test and train sets, keeping 30% of the data as the test set and 70% as the training set.

X_train, X_test, y_train, y_test = train_test_split(X1_pca, y, test_size=0.3, random_state=42)


# In[22]:


# Training the logistic regression model

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[23]:


# Making predictions

y_pred = log_reg.predict(X_test)


# In[24]:


# Evaluating the model:

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing model results:

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


# In[ ]:




