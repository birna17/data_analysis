#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[12]:


NAME = "Birna Ósk Valtýsdóttir"
COLLABORATORS = ""


# ---

# ## Linear algebra (10 points)

# In[13]:


from math import sqrt
from numpy import *


# #### Exercise 1 (1 point) 
# Write a function `vector_len(v)` that calulates the length (magnitude) of vector `v`.

# In[14]:


def vector_len(v):
    return linalg.norm(v)


# In[15]:


# Test cases 
print(vector_len(array([3,0,4])) == 5)
print(vector_len(array([-4,3,8])) == sqrt(89))


# #### Exercise 2 (1 point) 
# Write a function `add_and_take_determinant(a, b)` which adds matrixes `a` and  `b` together and then returns the determinant of the resulting matrix.

# In[16]:


def add_and_take_determinant(a,b):
    x = a+b
    return linalg.det(x)


# In[17]:


# Test case
a = array([[2, 5], [-1, 6]])
b = array([[-3, 2], [4, 4]])
add_and_take_determinant(a,b) == -31


# #### Exercise 3 (1 point) 
# Write a function `vector_dot(a, b)` that calulates the dot product of vectors `a` and  `b`. Do this manually without using numpy.dot().

# In[22]:


def vector_dot(a,b):
    dot = (a*b).sum()
    return dot

# dot = 0
# for i,j in zip(a,b):
#     dot = dot+i*j
# return dot


# In[23]:


# Test cases 
print(vector_dot(array([3,2,3]),array([1,6,4])) == 27 )
print(vector_dot(array([-8,3,5]),array([3,2,-7])) == -53)


# #### Exercise 4 (1 point) 
# Wrrite a function `cross_product(a,b)` that calculates the cross product of two three dimensional vectors `a` and `b` manually (without using numpy.cross).  

# In[24]:


def cross_product(a,b):
    c = [a[1]*b[2] - a[2]*b[1], 
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c


# In[25]:


# Test cases 
print(array_equal(
    cross_product(array([4,2,3]),array([6,1,5])),
    array([ 7, -2, -8]))
)

print(array_equal(
    cross_product(array([1,2,-4]),array([-1,2,0])),
    array([ 8, 4, 4]))
)


# #### Exercise 5 (2 points) 
# You go to the university store and purchase a soda along with two sandwiches. For this you pay 1600 ISK. The day after, you are studying with your friends when you decide to go to the university store again to pick up some lunch for the group. You buy three sodas and five sandwiches for the group to share which you pay 4100 ISK total for. How much does a soda cost and how much does a sandwich cost.
# 
# Solve this problem by writing this as a linear system before solving it using linalg.solve()

# In[38]:


a = array([[1, 2], [3, 5]])
b = array([1600, 4100])
eq = linalg.solve(a,b)
soda, sandw = eq[0], eq[1]

print("One soda costs: " + str('{:.0f}'.format(soda)) + ' ISK')
print("One sandwich costs: "+ str('{:.0f}'.format(sandw)) + ' ISK')


# #### Exercise 6 (2 points) 
# We can use a transformation matrix to rotate a vector with matrix multiplication. Write a function `rotate(v, degrees)` that uses a transformation matrix to rotate a two dimensional vector `v` counterclockwise around origin by the number of degrees specified by the `degrees` argument.

# In[50]:


def rotate(v, degrees):
    theta = radians(degrees)
    c, s = cos(theta), sin(theta)
    rot_matrix = array([[c, -s], [s,c ]])
    m = dot(rot_matrix, v)
    return m


# In[51]:


# Test cases 
print([round(number) for number in rotate(array([0, 1]), 90)] == [-1.0, 0.0])
print([round(number) for number in rotate(array([3, -1]), 180)] == [-3.0, 1.0])


# #### Exercise 7 (2 points) 
# Write a function `kirchhoff(G)` that implements [Kirchhoff's Theorem](https://en.wikipedia.org/wiki/Kirchhoff's_theorem) and returns the number of spanning trees for a graph. Kirchoffs Theorem relies on constructing the [Laplacian Matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) using the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) `G` for the graph. The Laplacian matrix  is equal to the difference between the graph's degree matrix (a diagonal matrix with the vertex degrees on the diagonal) and the adjacency matrix. After creating the Laplacian matrix, construct a Matrix Q* by by deleting any row and any column from the Laplacian matrix. The determinant of this new matrix equals the number of spanning trees for the graph. 
# 
# The input to the function will be the `G`, which is the adjacency matrix for the graph that you should calculate spanning trees for.

# In[75]:


# import networkx as nx
# G2 = nx.from_numpy_matrix(G)
# lap = nx.laplacian_matrix(G2).toarray()

def kirchhoff(G):
    D = diag(sum((G), axis=1)) #degree matrix, diagonal matrix
    L = D - G #Laplacian matrix, difference of D and G
    L = delete(L, 0, 0)
    L = delete(L, 0, 1)

    return abs(round(linalg.det(L)))


    



# In[76]:


# Test case
# The input that is given here (G) is the adjacency matrix for the diamond graph
# that shown in the wiki article, therefore the result should be 8, like the wiki article shows
round(kirchhoff(array([[0,1,1,0],
                      [1,0,1,1],
                      [1,1,0,1],
                      [0,1,1,0]]))) == 8 


# In[ ]:




