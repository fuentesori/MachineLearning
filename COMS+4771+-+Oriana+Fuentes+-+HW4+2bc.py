
# coding: utf-8

# In[1]:

from scipy.io import loadmat
import numpy as np


# In[2]:

hw4 = loadmat('hw4data.mat')
data = hw4['data']
labels = hw4['labels']


# In[3]:

size = 4096
b0 = np.array([0])
b = np.zeros(data.shape[1])
x = data[0:size]
iters = 10
obj = 0.65064
Y = labels[0:size]
B = np.concatenate([b,b0])
X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)


# In[4]:

def gradDescent(b0, b, iters, obj, x, Y):
    a=0
    B = np.concatenate([b,b0])
    X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)
    func = funccalc(B,X,Y)
    while func[0] > obj:
    #for j in range(iters):
        n=1
        temp = funccalc(B,X,Y)
        while temp >= func:
            A=B-gradientcalc(B,X,Y,n)
            temp = funccalc(A,X,Y)
            n=n/2
        B=B-gradientcalc(B,X,Y,n)
        func = funccalc(B,X,Y)
        a=a+1
    print(a)
    return func


# In[5]:

def gradientcalc(B,X,Y,n):
    gradient = (n*(np.sum(((X*np.vstack(np.exp(np.inner(B,X))))/np.vstack((1+np.exp(np.inner(B,X)))))-(X*Y), axis=0))/X.shape[0])
    return gradient


# In[6]:

def funccalc(B,X,Y):
    func = (np.sum(np.vstack(np.log((1+np.exp(np.inner(B,X))))) - Y*np.vstack((np.inner(B,X))), axis=0))/X.shape[0]
    return func


# In[7]:

print(gradDescent(b0, b, iters, obj, x, Y))


# In[8]:

#statistics on data
print("mean ", np.mean(data, axis=0))
print("stdev", np.std(data, axis=0))
print("max  ", np.max(data, axis=0))
print("min  ", np.min(data, axis=0))


# In[9]:

datanorm = (data - np.mean(data, axis=0))/(np.std(data, axis=0))
x = datanorm[0:size]
X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)
print(gradDescent(b0, b, iters, obj, x, Y))


# In[ ]:



