#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[16]:


NAME = "Birna Ósk Valtýsdóttir"
COLLABORATORS = ""


# ---

# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud


# ## Basic plotting (8 points)

# #### Exercise 1 (1 point)
# Plot the function $y = 2x+8$ on the interval -10 to 10

# In[37]:


x = np.linspace(-10,10) #interval
y = 2*x + 1   #function

plt.plot(x,y,'-r', label="y=2x+8")
plt.legend()
plt.grid()


# For the next few exercises we will be using the weather dataset imported in the cell below. The dataset includes hourly weather measurements along with weather descriptions.

# In[19]:


weather = pd.read_csv('weather.csv', index_col=0)
weather.head()


# #### Exercise 2 (2 points)
# Plot a boxplot that visualizes how the temperature `Temp (C)` variable is distributed against the categorical weather description variable `Weather`. Only include weather categories with values that occur at least 150 times. Make sure that the weather descriptions on the X axis are rotated by 45 degrees to ensure that they don't collide due to their length.

# In[20]:


w = weather.groupby('Weather').filter(lambda x: len(x) > 150) 

w.boxplot(by='Weather', column=['Temp (C)'], rot=45, grid=False, figsize=(10,10))

    


# #### Exercise 3 (2 points)
# Plot the same data as in exercise 3 (temperature against weather descriptions that occur at least 150 times) but this time as a violinplot. Make sure that the weather descriptions on the X axis are rotated by 45 degrees this time as well.

# In[21]:


w = weather.groupby('Weather').filter(lambda x: len(x) > 150) 
# x = w['Weather']
# y = w['Temp (C)']
# fig, ax = plt.subplots()

# ax.violinplot(y)

# w.violinplot(by='Weather', column=['Temp (C)'], rot=45, grid=False, figsize=(10,10))

ax = sns.violinplot(x = "Weather", y ="Temp (C)", data = w)
ax.tick_params(axis='x', rotation=45)


# #### Exercise 4 (1 point)
# Plot a [pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot) to visualize the releationship between the variables in the weather dataset.

# In[22]:


sns.pairplot(data=weather)


# #### Exercise 5 (2 points)
# Plot the functions $y = 3x-3$ and $y = -x^2 + 5$ on the interval -10 to 10. The functions should be plotted on the same figure as separate subplots.

# In[49]:


# plot1 = plt.subplot2grid((3,3) , (0,0), colspan=2)
# plot2 = plt.subplot2grid((3,3), (0,2), rowspan=3, colspan=2)
# x = np.arange(1,10)
# y1 = 3*x - 3
# y2 = -x**2 + 5
# plot1.plot(x,y1, label='y = 3*x - 3')
# plot1.set_title('y = 3*x - 3')

# plot2.plot(x,y2, label='y = -x**2 + 5')
# plot2.set_title('y = -x**2 + 5')

# plt.tight_layout()
# plt.show()

x = np.linspace(-10,10,100) #interval
y1 = 3*x - 3
y2 = -x**2 + 5
fig, ax= plt.subplots(2)

ax[0].plot(x,y1, label='y = 3*x - 3')
ax[0].legend()
ax[0].grid()
ax[1].plot(x,y2,'-r', label='y = -x**2 + 5')

plt.legend()
plt.grid()


# ## Advanced plotting (12 points)

# #### Exercise 6 (2 points)
# Go to [this cool website](http://boundingbox.klokantech.com/) and draw a boundary box around Iceland. Change the settings in the lower left corner to *DublinCore* for a more readable format of the boundaries. Copy the boundary coordinates. Using the Matplotlib `basemap` package, create a `Basemap` object using the information you gathered in the previous step. Pick any projection you like. Fill the land part with a grey-ish color and the water with a blue-ish color and draw the coastline of Iceland.

# In[26]:


m = Basemap(projection='merc',
            llcrnrlon=-24.92,
            llcrnrlat=62.92,
            urcrnrlon=-13.08,
            urcrnrlat=66.85,
           resolution='l')
#m.etopo()
# m.bluemarble()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='#CCCDC6')
m.drawcoastlines(color = '0.15')
plt.title('Iceland')
plt.show()

# westlimit=-24.92; southlimit=62.92; eastlimit=-13.08; northlimit=66.85


# #### Exercise 7 (3 points)
# Use the [earthquake API at apis.is](http://apis.is/earthquake/is) to obtain information about earthquakes in iceland on a json format. Then create a pandas dataframe from that information.
# 
# There are of course many ways to do this but it is recommended to use the [requests](https://docs.python-requests.org/en/latest/) library to fetch the data and the json_normalize function from pandas to create the dataframe.

# In[29]:


re = requests.get('https://apis.is/earthquake/is', auth=('user', 'pass'))
r = re.json()
df = pd.concat([pd.DataFrame(v) for k,v in r.items()], keys=r)
df


# #### Exercise 8 (4 points)
# Using Basemap, plot the coastline of Iceland again and plot every earthquake from the dataframe created in the previous exercise as a filled circle with the size of the earthquake represented by the size of the circle.

# In[56]:


m = Basemap(projection='merc',
            llcrnrlon=-24.92,
            llcrnrlat=62.92,
            urcrnrlon=-13.08,
            urcrnrlat=66.85,
           resolution='l')

m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='#CCCDC6')
m.drawcoastlines(color = '0.15')

longitudes = df['longitude'].tolist()
latitudes = df['latitude'].tolist()
x,y = m(longitudes, latitudes)

m.scatter(x,y, c="red",s=3*df["size"]**2)

plt.title('Earthquakes in Iceland')
plt.show()


# #### Exercise 9 (3 points)
# Use the [wordcloud](https://amueller.github.io/word_cloud/) library to plot a wordcloud with the weather descriptions from the weather dataset. The size of the descriptions in the wordcloud should represent their frequency in the dataset.

# In[73]:


text = " ".join(i for i in weather.Weather.astype(str))

wordcloud = WordCloud(relative_scaling=.4).generate(text)

plt.axis("off")

plt.figure(figsize=[40,20])

plt.tight_layout(pad=0)

plt.imshow(wordcloud, interpolation='bilinear')

plt.show()


# In[ ]:




