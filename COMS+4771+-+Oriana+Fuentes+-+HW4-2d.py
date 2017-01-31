
# coding: utf-8

# In[1]:

import math
from scipy.io import loadmat
import numpy as np


# In[2]:

hw4 = loadmat('hw4data.mat')
data = hw4['data']
labels = hw4['labels']


# In[3]:

totaldata = 4096
size = math.floor(totaldata*0.80)
holdout =  totaldata-size
b0 = np.array([0])
b = np.zeros(data.shape[1])
x = data[0:size]
xH = data[size+1:totaldata]
iters = 10
obj = 0.65064
Y = labels[0:size]
YH = labels[size+1:totaldata]
B = np.concatenate([b,b0], axis=0)
B = B.reshape(B.shape[0],1)
X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)
xH = np.concatenate([xH, np.vstack(np.ones(xH.shape[0]))], axis=1)


# In[4]:

def gradDescent(b0, b, iters, obj, x, Y, xH, YH):
    a=0
    err = 0
    error = 0
    stop =False
    B = np.concatenate([b,b0])
    X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)
    func = funccalc(B,X,Y)
    while not(stop==True and a>32):
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
        if np.mod(np.log2(a),1)==0:
            errortemp = testholdout(B, xH, YH)
            err = err+1
            if (errortemp >(0.99*error)) and err>0:
                stop=True
            error = errortemp            
    print("iterations", a,"objective function:", func,"holdout error rate:",error)
    return func


# In[5]:

def gradientcalc(B,X,Y,n):
    gradient = (n*(np.sum(((X*np.vstack(np.exp(np.inner(B,X))))/np.vstack((1+np.exp(np.inner(B,X)))))-(X*Y), axis=0))/X.shape[0])
    return gradient


# In[6]:

def testholdout(B,xH, YH):
    preds = (np.sign(xH.dot(B))+1)/2
    error = (np.count_nonzero(np.abs(np.vstack(preds)-YH)))/YH.shape[0]
    return error


# In[7]:

def funccalc(B,X,Y):
    func = (np.sum(np.vstack(np.log((1+np.exp(np.inner(B,X))))) - Y*np.vstack((np.inner(B,X))), axis=0))/X.shape[0]
    return func


# In[8]:

print(gradDescent(b0, b, iters, obj, x, Y, xH, YH))


# In[9]:

#statistics on data
print("mean ", np.mean(data, axis=0))
print("stdev", np.std(data, axis=0))
print("max  ", np.max(data, axis=0))
print("min  ", np.min(data, axis=0))


# In[10]:

datanorm = (data - np.mean(data, axis=0))/(np.std(data, axis=0))
x = datanorm[0:size]
xH = datanorm[size+1:totaldata]
X = np.concatenate([x, np.vstack(np.ones(x.shape[0]))], axis=1)
xH = np.concatenate([xH, np.vstack(np.ones(xH.shape[0]))], axis=1)
print(gradDescent(b0, b, iters, obj, x, Y, xH, YH))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



