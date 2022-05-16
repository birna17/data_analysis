#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[8]:


NAME = "Birna Ósk Valtýsdóttir"
COLLABORATORS = ""


# ---

# ## Statistics (20 points)

# In[282]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from scipy import stats
import math
from scipy import stats
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import power


# #### Exercise 1 (2 points)
# Due to the COVID-19 traveling restrictions, Icelandair is facing difficulties with the number of empty seats in each flight. For the last flight of the month, there are still 16 empty seats. Past experience data shows that 80% of the ticket buyers show up for the flight. 
# 
# a) If the marketing team decides to sell 20 additional tickets, what is the probatility of overbooking the flight? <br>
# b) What is the probability of having at least one empty seat if 20 additional tickets are sold?
# 
# *Hint: You can use the function binom.cdf to compute the cumulative distribution function for a binomial variable.* 

# In[287]:


n = 20
x = 17
p = 0.8
a = stats.binom.cdf(x,n,p)

x2 = 15
b = stats.binom.cdf(x2,n,p)


print('a) ' + str('{:.2%}'.format(a)) + '\tb) ' + str('{:.2%}'.format(b)))


# #### Exercise 2 (3 points)
# Suppose that the number of lunch meals sold by Málið every day has a normal distribution with a mean of 87 and a standard deviation of 22 lunch meals. Then a random sample consisting of 16 days is taken to estimate the average number of daily meals. 
# 
# a) What is the standard error of the sample mean? <br>
# b) What is the probability that the sample mean is less than 100 meals?<br>
# c) What is the probability that the sample mean is less than 85 meals but more than 95 meals?

# In[284]:


mean = 87
s = 22
n = 16

a = s/math.sqrt(n)

# b = z score
z1 = (100-mean)/a
b = stats.norm.cdf(z1)

# c = OR more than .. plusar saman 
z2 = (85-mean)/a
z3 = (95-mean)/a

c = stats.norm.cdf(z2) + (1 - stats.norm.cdf(z3))

print('a) ' + str(a) + '\tb) ' + str('{:.2%}'.format(b)) + '\tc) ' + str('{:.2%}'.format(c)))


# #### Exercise 3 (0.5 points)
# Read the csv document `Spotify-2000.csv` into a pandas data frame. The dataset contains audio statistics for the top 2000 tracks on spotify (out of tracks released between 1956 and 2019).

# In[285]:


spot = pd.read_csv('Spotify-2000.csv')


# #### Exercise 4 (2 points) 
# Create a sample with 25 `Popularity` scores. Using R=1000 calculate the 95% bootstrap confidence interval for the mean. Output the mean of the bootstrap along with the confidence interval.

# In[286]:


#create random sample from spot['Popularity']

sample25 = resample(spot['Popularity'], n_samples=25, replace=False)
mean = sample25.mean()

print('Mean: ' + str(mean))

results = []
for nrepeat in range(1000):
    sample = resample(sample25)
    results.append(sample.mean())
results = pd.Series(results)

confidence_interval = list(results.quantile([0.05, 0.95]))

# confidence_interval2 = stats.t.interval(alpha=0.95, df=len(sample25)-1, loc=np.mean(sample25), scale=stats.sem(sample25))

print('Confidence interval: ' + str(confidence_interval))
# print('Confidence interval: ' + str(confidence_interval2))


# #### Exercise 5 (0.5 points)
# Create a new dataframe that contains only the songs that belong to the `alternative rock`, `dance pop`, and `dutch pop` genres. **Use this dataframe for all of the remaining exercises in the assignment**

# In[24]:


li = ['alternative rock','dance pop','dutch pop']

sp = spot.iloc[np.where(spot['Top Genre'].isin(li))]

sp


# #### Exercise 6 (3 points)
# Plot the histograms, boxplots, and Q-Q plots of `Popularity` split by `Top Genre`. 

# In[298]:


g1 = sp[sp['Top Genre'] == 'alternative rock']
g1_pop = [i for i in g1['Popularity']]

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(g1_pop, bins=3, rwidth=0.6)
ax.set_ylabel('Popularity')
ax.set_xlabel('alternative rock')
plt.suptitle('')
plt.show()

g2 = sp[sp['Top Genre'] == 'dutch pop']
g2_pop = [i for i in g2['Popularity']]

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(g2_pop, bins=3, rwidth=0.6)
ax.set_ylabel('Popularity')
ax.set_xlabel('dutch pop')
plt.suptitle('')
plt.show()

g3 = sp[sp['Top Genre'] == 'dance pop']
g3_pop = [i for i in g3['Popularity']]

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(g3_pop, bins=3, rwidth=0.6)
ax.set_ylabel('Popularity')
ax.set_xlabel('dance pop')
plt.suptitle('')
plt.show()

# sp.hist(data,bins=3, by='Top Genre', figsize=(10,10))
# fig, ax = plt.subplots(figsize=(6,6))
# ax.hist(sp[', bins=3)
# ax.set_xlabel('Top Genres')

ax = sp.boxplot(by='Top Genre', column=['Popularity'], grid=False, figsize=(8,8))
ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('Popularity')
plt.suptitle('')

a = sm.qqplot(sp['Popularity'][sp['Top Genre'] == 'alternative rock'], stats.norm, fit=True, line='45')
b = sm.qqplot(sp['Popularity'][sp['Top Genre'] == 'dutch pop'], stats.norm, fit=True, line='45')
c = sm.qqplot(sp['Popularity'][sp['Top Genre'] == 'dance pop'], stats.norm, fit=True, line='45')


# #### Exercise 7 (1 point)
# Based on the plots, draw your first conclusions about the differences in the `Popularity` scores for the different  `Top Genre` values.

# Based on the plots, the scores are normally distributed. Looks like the null hypothesis will be rejected since that the mean between the three populations is different. 

# #### Exercise 8 (2 points)
# Implement an ANOVA over `Popularity` and `Top Genre`. Interpret the `p-value` obtained. 

# In[297]:


stat, pvalue = stats.f_oneway(g1_pop, g2_pop, g3_pop)

print(f'F-Statistic: {stat / 2:.4f}')
print(f'p-value: {pvalue / 2:.36f}')


# With the p-value being so close to 0 it is therefore statistically significant. The null hypothesis is therefore rejected. 


# #### Exercise 9 (2 points)
# Use a t-Test to evaluate for statistically significant differences in the `Popularity` score between `dance Pop` and `dutch pop` genres.

# In[296]:


# print(np.var(g3_pop), np.var(g2_pop))

# 157.11714327188272/148.36350723140495 = 1.0590012746653703 is less than 4, so i can assume that the population variances are equal

res = stats.ttest_ind(a=g3_pop, b=g2_pop, equal_var=True)

print(f'p-value for single sided test: {res.pvalue / 2:.36f}')


# #### Exercise 10 (2 points)
# Use a t-Test to evaluate for statistically significant differences in the `Popularity` score between `dance Pop` and `alternative rock` genres.

# In[276]:


# tstat, pvalue, df = sm.stats.ttest_ind(g3_pop, g2_pop, usevar='unequal', alternative='smaller')

# print(f'p-value: {pvalue:.4f}')

res = stats.ttest_ind(a=g3_pop, b=g1_pop, equal_var=True)

print(f'p-value for single sided test: {res.pvalue / 2:.4f}')


# #### Exercise 11 (2 point)
# Interpret the `p-values` obtained in Exercises 9 and 10. Elaborate on whether or not those values confirm your answer in Exercise 7.

# Yes they do confirm my answer in ex_7. The p-values are both closer to 0 than they are to 1, it is statistically significant and the null hypothesis is rejected and we accept the alternative. Normal distribution is at hand and everything seems to be in order. 
