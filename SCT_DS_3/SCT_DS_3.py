#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df = pd.read_csv("bank-full.csv" , sep=";")
df.head()


# In[11]:


print("="*60)
print("Dataset Shape")
print(df.shape)

print("\nDataset Information")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
display(df.describe(include="all"))


# In[12]:


print("Duplicate Records :", df.duplicated().sum())

df.drop_duplicates(inplace=True)


# In[14]:


plt.figure(figsize=(5,3))

sns.countplot(x="y", data=df)

plt.title("Target Variable Distribution")

plt.show()


# In[15]:


plt.figure(figsize=(5,3))

sns.histplot(df["age"], bins=30, kde=True)

plt.title("Customer Age Distribution")

plt.show()


# In[16]:


plt.figure(figsize=(5,3))

sns.countplot(
    y="job",
    data=df,
    order=df["job"].value_counts().index
)

plt.title("Job Distribution")

plt.show()



# In[17]:


plt.figure(figsize=(5,3))

sns.countplot(x="marital", data=df)

plt.title("Marital Status")

plt.show()


# In[18]:


plt.figure(figsize=(5,3))

sns.countplot(
    y="job",
    hue="y",
    data=df
)

plt.show()


# In[27]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)


# In[28]:


X = df.drop("y", axis=1)

y = df["y"]


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# In[30]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# In[33]:


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", round(accuracy*100,2),"%")


# In[34]:


print(classification_report(y_test, y_pred))


# In[35]:


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()


# In[37]:


from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.show()


# In[ ]:




