#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning and EDA on Titanic Datasets

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


dataset = sns.load_dataset("titanic")
dataset.head(3)


# In[4]:


dataset.describe()


# In[9]:


missing = pd.DataFrame({
    "Missing Values": dataset.isnull().sum(),
    "Percentage": (dataset.isnull().sum()/len(dataset))*100
})

missing = missing[missing["Missing Values"] > 0]

missing


# In[14]:


dataset["age"] = dataset["age"].fillna(dataset["age"].median())

dataset["embarked"] = dataset["embarked"].fillna(dataset["embarked"].mode()[0])

dataset["embark_town"] = dataset["embark_town"].fillna(dataset["embark_town"].mode()[0])

dataset.drop(columns="deck", errors = "ignore" , inplace=True)

dataset.isnull().sum()


# In[15]:


print("Duplicate Rows:", dataset.duplicated().sum())

dataset.drop_duplicates(inplace=True)


# In[19]:


plt.figure(figsize=(5,3))

sns.countplot(x="survived", data=dataset)

plt.title("Passenger Survival Count")

plt.show()


# In[21]:


plt.figure(figsize=(5,3))

sns.countplot(x="sex", data=dataset)

plt.title("Gender Distribution")

plt.show()


# In[23]:


plt.figure(figsize=(5,3))

sns.countplot(x="pclass", data=dataset)

plt.title("Passenger Class")

plt.show()


# In[25]:


plt.figure(figsize=(5,3))

sns.histplot(dataset["age"], bins=30, kde=True)

plt.title("Age Distribution")

plt.show()


# In[27]:


plt.figure(figsize=(5,3))

sns.histplot(dataset["fare"], bins=30, kde=True)

plt.title("Fare Distribution")

plt.show()


# In[28]:


plt.figure(figsize=(5,3))

sns.boxplot(x="survived", y="age", data=dataset)

plt.title("Age vs Survival")

plt.show()


# In[29]:


numeric_dataset = dataset.select_dtypes(include=np.number)

plt.figure(figsize=(6,4))

sns.heatmap(
    numeric_dataset.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.show()


# In[32]:


plt.figure(figsize=(6,4))

sns.pairplot(
    dataset[
        ["survived",
         "age",
         "fare",
         "pclass"]
    ],
    hue="survived"
)

plt.show()


# In[ ]:




