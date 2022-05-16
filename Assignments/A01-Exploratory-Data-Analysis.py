#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = ""
COLLABORATORS = ""


# ---

# ## Exploratory data analysis (20 points)
# In this exercise you will be practicing your skills in the first step of any data science project, exploratory data analysis. Using the pandas [documentation](https://pandas.pydata.org/docs/index.html) is recommended and the  `Practical Statistics For Data Scientists` book is very useful as well. This assignment contains 15 exercises which add up to 20 points. Note that the points on jupyterhub are not the same as on canvas, the jupyterhub points are only used to calculate your grade for the assignment. For example 17 points gives 17/20 = 0.85 which means a grade of 8.5 for the assignment. 

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import trim_mean


# #### Exercise 1 (1 point)
# Read the csv document [`bodyPerformance.csv`](./bodyPerformance.csv) into a `pandas` dataframe. Then use the pandas [head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) function to print out the first records in the dataset. The dataset comes from the `Korea Sports Promotion Foundation` and contains the results of various body measurements for individuals, along with their performance in various exercises.

# In[2]:


perf = pd.read_csv("body_performance.csv") 
perf.head()


# #### Exercise 2 (1 point)
# Output the number of **records** (rows) and **features** (columns) in the dataset.

# In[3]:


perf.shape


# #### Exercise 3 (1 point)
# Output the last 20 Records of the dataframe.

# In[4]:


perf.tail(20)


# #### Exercise 4 (1 point)
# Output the number of unique values for each of the variables in the dataset.

# In[82]:


perf.nunique()


# #### Exercise 5 (1 point)
# Output all unique values of the gender variable in the dataset.

# In[83]:


pd.unique(perf['gender'])


# #### Exercise 6 (2 points)
# For the following variables in the dataset determine their data type (continous, binary, etc.). Include an explanation for why each of them belongs to the data type determined.
# * body fat_%
# * gender
# * weight_kg
# * sit-ups counts

# In[ ]:





# Body fat: Continuous data. Body fat cant be lower than some x% or higher than some y%
# Gender: Binary data since it has only two categories of values, e.g. m/f
# Weight: Continuous data, can take any value between two numbers.. e.g. from 0 kg to 300 kg
# Sit-ups: Discrete data, int,since its being counted, there is no half sit-up. How ever if there was to be an estimate of sit-ups who are less than 1, it would have to be a float. 

# #### Exercise  7 (1 point)
# Output the records in the dataframe ordered by the `broad jump_cm` value in descending order so that the records with the highest value is displayed at the top.

# In[8]:


perf.sort_values(by=['broad jump_cm'], ascending=False)


# #### Exercise  8 (2 points) 
# Output the mean and the trimmed mean(5%) of the `sit-ups counts` variable. Also include an explanation for the difference mean and trimmed mean and how that is visible in the difference between the numbers.

# In[9]:


m = perf['sit-ups counts'].mean()
print('Mean: ', m)
print('Trimmed mean: ', trim_mean(perf['sit-ups counts'], 0.05))


# #### Exercise  9 (1 point) 
# Output the `body fat_%` values at the deciles (the 10th, 20th, ..., 90th percentiles).

# In[10]:


print('10th:', perf['body fat_%'].quantile(0.1))
print('20th:', perf['body fat_%'].quantile(0.2))
print('30th:', perf['body fat_%'].quantile(0.3))
print('40th:', perf['body fat_%'].quantile(0.4))
print('50th:', perf['body fat_%'].quantile(0.5))
print('60th:', perf['body fat_%'].quantile(0.6))
print('70th:', perf['body fat_%'].quantile(0.7))
print('80th:', perf['body fat_%'].quantile(0.8))
print('90th:', perf['body fat_%'].quantile(0.9))


# #### Exercise  10 (1 point) 
# Output the mean, standard deviation, minimum value, maxiumum value, count, 25th percentile, 50th percentile and 75th percentile for all variables in the dataset. If you look at the pandas documentation you will see that this can be achieved with a single function.

# In[11]:


perf.describe()


# #### Exercise  11 (1 point) 
# Output the median height for each gender. Use the pandas [groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) function in your solution.

# In[12]:


perf.groupby(['gender'])['height_cm'].median()


# #### Exercise  12 (2 points) 
# Next we want to determine out the average `gripForce` for females aged ≤ 20 but less than 50 years old. We want to compare that number to the average `gripForce` for females aged ≥ 50 years old.
# Output the following: 
# * Average gripForce for women aged ≤ 20 and < 50 years old.
# * Average gripForce for women older than 50 as well
# * The number of samples in each of these categories

# In[78]:


av1 = perf.loc[(perf['gender']=='F') & (perf['age']>= 20) & (perf['age']>50)]['gripForce']
av2 = perf.loc[(perf['gender']=='F') & (perf['age']<50)]['gripForce']
print(av1.mean())
print(av2.mean())
count = av1.count() + av2.count()
print(count)


# #### Exercise  13 (2 points) 
# Now lets visualize the relationship between the `height_cm` and `weight_kg` variables. Plot a scatter plot of the data where the x axis represents `height_cm` and the y axis represents `weight_kg`. The dots on the plot should be colored green.

# In[84]:


perf.plot.scatter('height_cm', 'weight_kg',c='Green',figsize=(10,10))


# #### Exercise  14 (1 point) 
# This time we are interested in knowing how the broad jump variable is distributed for males. Plot a histogram that visualizes the distribution of data for the `broad jump_cm` variable for records where the gender is Male. The histogram should consist of 15 equally sized bins. 

# In[81]:


m = perf.loc[perf['gender']=='M']
m['broad jump_cm'].plot.hist(15)


# #### Exercise  15 (2 points) 
# Output a correlation table that includes only the `age`, `body fat_%`, `diastolic_bp` and `systolic_bp` variables from the dataset.

# In[53]:


perf[['age','body fat_%', 'diastolic_bp', 'systolic_bp']].corr()


# In[ ]:




