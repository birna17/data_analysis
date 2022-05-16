#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Birna Ósk Valtýsdóttir"
COLLABORATORS = ""


# ---

# ## Data preprocessing and model evaluation (20 points)

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
from scipy.stats import zscore


# ### Data preprocessing (8 points)

# #### Exercise 1 (1 point)
# For the data preprocessing exercises we will be using an Olympic athlete dataset called `athletes.csv`. The dataset has been modified to include only athletes that competed in men's handball during the 2008 Olympics. Read the CSV file containing the dataset into a dataframe. 
# 
# Unfortunately, we are missing height and weight values for a few of our athletes so we will not be able to use their data. Find all rows where a height or weight value is missing and remove them from the dataset.

# In[3]:


players = pd.read_csv('athletes.csv').dropna(subset=['Weight', 'Height'])
players


# #### Exercise 2 (1 point)
# There seems to have been some mistake when the data was entered and it looks like some players were duplicated in the data. Remove all duplicate entries from the data to ensure that each player only occurs once.

# In[4]:


# ath_cols = ['Name', 'Age', 'Height', 'Weight', 'NOC', 'Year', 'Country', 'Medal']
# players = players[ath_cols].drop_duplicates()
# players = players.reset_index(drop=True)
# players['player_id'] = players.index
players = players.drop_duplicates()
players


# #### Exercise 3 (1 point)
# The NOC column contains the three letter code that is used to identify a National Olympic committee. However, it looks like the values for the NOC code might have been pulled from different sources, since the NOC code seems to be in multiple different formats. This is not ideal since we need to use that column when joining with another dataframe later. Fix the formatting of the values in the NOC column so that they all consist of three capital letters with no extra characters.

# In[5]:


players = players.replace({'NOC': {'F-R-A': 'FRA', '[ESP]': 'ESP', '(RUS)': 'RUS'}})
pd.unique(players['NOC'])
players


# In[6]:


players['NOC'].unique()


# #### Exercise 4 (1 point)
# The NOC  column is not descriptive enough for us. Now that we have fixed it, we want to join it with another dataframe that contains more information. Read the CSV file `noc_regions.csv` into a dataframe. Then merge that dataframe with the athletes dataframe to make sure that the new athlete dataframe includes the full region name. Afterwards, move the region name column, so that is the next column following the NOC column and make sure that the resulting dataset only has one NOC code column.

# In[7]:


dff = pd.read_csv('noc_regions.csv')

players = players.merge(dff,how='left', left_on='NOC', right_on='NOC_code')

players = players[['Name', 'Age', 'Height', 'Weight', 'NOC', 'region', 'Year', 'Country', 'Medal']]
# players = players.rename(columns={'region': 'Region'})
players


# #### Exercise 5 (1 point)
# The dataframe includes a variable that describes which medal a player won, if any. We are interesting in knowing which players   are medal winners in general and which are not. Add a `Medal_winner` column to the dataframe that has a True/False value for each player where the value depends on whether that player won a medal or not.

# In[8]:


# if_won = [True if i is None else False for i in players['Medal']]

# players['Medal_Winner'] = if_won
# players


players.loc[players['Medal'].isnull(),'Medal_winner'] = False
players.loc[players['Medal'].notnull(), 'Medal_winner'] = True

players


# #### Exercise 6 (2 points)  
# The individuals responsible for entering the data seem to have made some more mistakes. By looking at the data we can see that there are a few obvious outliers that need to be fixed. Find the obvious outliers in the data using z-scores and fix them in the way that you see fit (remove the row or replace the value). Note that the outliers are only included in variables that have numerical values. In the cells below you should show both the code used to find the outliers and the code used to fix them.

# In[9]:


agemean=players['Age'].mean()
agestd=players['Age'].std()
threshold = 3
outlierage = [] 
for i in players['Age']: 
    z = (i-agemean)/agestd 
    if z > threshold: 
        outlierage.append(i) 
print('Outliers in Age are', outlierage) 

hmean=players['Height'].mean()
hstd=players['Height'].std()
threshold = 3
outlierh = [] 
for i in players['Height']: 
    z = (i-hmean)/hstd 
    if z > threshold: 
        outlierh.append(i) 
print('Outliers in Height are', outlierh) 

wmean=players['Weight'].mean()
wstd=players['Weight'].std()
threshold = 3
outlierw = [] 
for i in players['Weight']: 
    z = (i-wmean)/wstd 
    if z > threshold: 
        outlierw.append(i) 
print('Outliers in Weight are', outlierw) 

outliers = outlierage + outlierh + outlierw

# for k, v in players.items():
#     if v['Age'] in outliers:


players = players.replace({'Age': {250: agemean}})
players = players.replace({'Height': {298.0: hmean}})
players = players.replace({'Weight': {350.0: wmean}})
players


# #### Exercise 7 (1 point)
# Not all classifiers support categorical variables. We want to feed our data to one of these classifiers and we believe that it would be beneficial to include the region column. This means that we have to use One-hot encoding to encode the categorical region variable as multiple binary variables. Create a new dataframe where One-hot encoding has been used to encode the region variable as multiple binary variables.

# In[10]:


# print(players['region'])
onehot = pd.get_dummies(players['region'], columns = ['dummy_col', 'dummy_col2'])

players = players.drop('region', axis=1)
players = players.join(onehot)
#transform data 
# onehot = encoder.fit_transform(data)
players


# ## Model evaluation (12 points)
# For the model evaluation exercises we will be using the wine_quality dataset which is a imbalanced dataset from imblearn. It is a dataset that contains wine information and it splits wines up into two classes. The target class 1 for bad wines (wines with a rating >=4) and another class -1 that consists of good wines. 
# 
# The classifier that we will use for these exercises is the [random forest classifier implemenation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from sklearn. Note that the algorithm hasn't been covered yet in the course, so you don't need to fully understand it for these exercises. You can treat it as a "black box" and focus on interpreting the results.

# In[4]:


wine_quality = fetch_datasets()["wine_quality"]
x = wine_quality.data
y = wine_quality.target


# #### Exercise 8 (1 point)
# Output the number of values that belong to each of the classes in the dataset.

# In[5]:


l = [True if i==1 else False for i in y]
bad = l.count(True)
good = l.count(False)

print('Target Class 1: ', bad)
print('Other Class -1: ', good)


# #### Exercise 9 (1 point)
# Split the data from the wine_quality dataset into training and test data. The split should be 75% training data and 25% test data. Name the training data input `X_train`, the output `y_train`, the test data input `X_test` and the test data output `y_test`

# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .25, random_state= 100)
print('The size of the training set is:', x_train.shape, 'with the target vector: ', y_train.shape)
print('The size of the test set is:', x_test.shape, 'with the target vector: ', y_test.shape)


# #### Exercise 10 (1 point)
# In the cell below a random forest classifier is trained using the training data. Measure your accuracy using the test data and output it.

# In[7]:


clf = RandomForestClassifier()
clf.fit(x_train, y_train)


# In[8]:


from sklearn.metrics import accuracy_score
# x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
accuracy = accuracy_score(y_test,clf.predict(x_test))
# def Performance(y_test,X_test):
print('Accuracy: ', accuracy)
#     print('Recall: ',recall_score(y_test,clf.predict(x_test)))
#     print('Precision: ',precision_score(y_test,clf.predict(x_test),zero_division=0))
# #     print('AUC: ',roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
    
# Performance(y_test, x_test)


# #### Exercise 11 (1 point)
# Output the precision and recall for the model created above using the test data.

# In[9]:


recall = recall_score(y_test,clf.predict(x_test))
precision = precision_score(y_test,clf.predict(x_test),zero_division=0)
print('Recall: ',recall)
print('Precision: ',precision)


# #### Exercise 12 (1 point)
# Output the confusion matrix for the model created above using the test data. We are more interested in the bad wines so make sure that the confusion matrix displays the bad wines as positives (on the left hand side of the matrix).

# In[13]:


CM=confusion_matrix(y_test , clf.predict(x_test),labels=[1,-1])
print(CM)


# #### Exercise 13 (1 point)
# 
# We are especially interested in being able to accurately predict the bad wine class. However the bad wine class is very underrepresented so we need to balance the training data so we can train a new model that is better at finding examples that belong to the bad wine class.
# Use SMOTE oversampling to balance the training data so that the under-represented class with the bad wines has as many examples as the other one.

# In[15]:


sm = SMOTE(random_state=42)
x_smote, y_smote = sm.fit_resample(x, y)


# #### Exercise 14 (2 points)
# Train a new model using the newly balanced training dataset (you can use the classifier training code from above as a reference, just remember to use the new oversampled training data). Then calculate and print the accuracy, precision, recall and confusion matrix for this new model using your test data. Make sure again that the confusion matrix displays the bad wines as positives.

# In[29]:


accuracy = accuracy_score(y_smote,clf.predict(x_smote))
recall = recall_score(y_smote,clf.predict(x_smote))
precision = precision_score(y_smote,clf.predict(x_smote),zero_division=0)

print('Accuracy: ', accuracy)
print('Recall: ',recall)
print('Precision: ',precision)

cm =confusion_matrix(y_smote , clf.predict(x_smote),labels=[1,-1])
print(cm)


# #### Exercise 15 (2 points)
# 
# Compare the metrics of the model trained on the original data and the model trained on the oversampled data. What is the biggest difference? Why did the performance change in the way it did with the second model.

# well the biggest difference is obviously the precicion, which goes from being only 53% over to being 99.5% 

# #### Exercise 16 (2 points)
# Plot a precision-recall curve for each of the models created.

# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
lr_pr, lr_re, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
plt.plot(lr_pr, lr_re, color='orange', label='Logistic')
# plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Original')
plt.legend()
plt.show()
    

lrp, lre, thresh = precision_recall_curve(y_smote, clf.predict(x_smote))
plt.plot(lrp, lre, color='orange', label='Logistic')
# plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Rebalanced')
plt.legend()
plt.show()
    


# In[ ]:




