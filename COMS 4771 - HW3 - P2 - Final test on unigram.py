
# coding: utf-8

# In[9]:

from __future__ import division
from scipy.io import loadmat
import numpy as np
import scipy as sci
from scipy import stats
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
import time
import pandas as pd
import sklearn as sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import KFold


# In[10]:

start_time = time.time()
#runing with 200,000 first rows of training data
samplerows = 200000 #200000
testrows = 200000 #320122
validations =5
methods =1
data = pd.read_csv('reviews_tr.csv', nrows=samplerows, iterator=True)
tdata = pd.read_csv('reviews_te.csv', nrows=testrows, iterator=True)

#split data into text and labels
dlabels = data['label']
dtext = data['text']
tdlabels = tdata['label']
tdtext = tdata['text']


# In[11]:

#set up representations
#unigram
vectorizer = CountVectorizer(min_df=1)


# In[12]:

#unigram representation
uni_dtext = vectorizer.fit_transform(dtext)
dictuni = vectorizer.get_feature_names()

print('Representations')
print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:

def Perceptron (pdata, ptdata, plabels, ptlabels):
#perceptron function
    preds = []

    w = np.zeros(pdata.shape[1])
    u = np.zeros(pdata.shape[1])
    b = 0
    beta = 0
    c = 0
    
    index = np.arange(0,(pdata.shape[0]))
    np.random.shuffle(index)

    for i in index:
        if (plabels[i]==0):
            y=-1
        else:
            y=1
        x = (pdata[i]).toarray()
        test = (y*(np.inner(w,x)+b))
        if ((y*(np.inner(w,x)+b))< 0) or ((y*(np.inner(w,x)+b))== 0):
            w = w + (y*x)
            b = b +y
            u = u + (y*c*x)
            beta = beta + (y * c)
        c = c +1

    u = np.zeros(pdata.shape[1])
    b = 0
    beta = 0
    c = 0        
        
    np.random.shuffle(index)

    for i in index:
        if (plabels[i]==0):
            y=-1
        else:
            y=1
        x = (pdata[i]).toarray()
        test = (y*(np.inner(w,x)+b))
        if ((y*(np.inner(w,x)+b))< 0) or ((y*(np.inner(w,x)+b))== 0):
            w = w + (y*x)
            b = b +y
            u = u + (y*c*x)
            beta = beta + (y * c)
        c = c +1

    finalw = w - (1/c*u)
    finalbeta = b - (1/c*beta)


    #test perceptron
    for i in range(ptdata.shape[0]):
        x = (ptdata[i]).toarray()
        newlabel = np.inner(finalw,x)
        if newlabel <0 or newlabel == 0:
            newlabel = 0
        else:
            newlabel = 1

        preds.append(newlabel)
    preds = np.array(preds)


    error = check_error(preds,ptlabels)
    return error


# In[14]:

#subroutines for Naive Bayes
def create_priors(priorslabels,classes):
    indicesp = [a for a, x in enumerate(priorslabels) if x in [1]]
    priors = [(len(priorslabels[indicesp]))/len(priorslabels)]
    for i in range(0,classes-1):
        indicesp = [a for a, x in enumerate(priorslabels) if x in [i]]
        subprior = (len(priorslabels[indicesp]))/len(priorslabels)
        priors.append(subprior)
    priors= csr_matrix(np.log(np.array(priors)))
    return priors

def create_mus(usedata,uselabels,classes):
    #create matrix of laplace smoothed mus - creates a 20 x 60k
    indices = [a for a, x in enumerate(uselabels) if x in [0]]
    collapse = usedata[indices].sum(axis=0)
    collapse = csc_matrix((collapse +1)/(2+len(indices)))
    for i in range(1,classes):
        indices = [a for a, x in enumerate(uselabels) if x in [i]]
        subdata1 = usedata[indices].sum(axis=0)
        subdata1 = csc_matrix((subdata1 +1)/(2+len(indices)))
        collapse = vstack([csc_matrix(collapse), subdata1],format="csc")
    return collapse

def model(collapse,datamodel,priorsmodel,classes):
    #create matrix of (log(1-mus)) - creates a 20 x 60k
    minusmu = csc_matrix(np.log(1-(collapse.toarray())))
    #create matrix of (logmu - log (1-mu)) - creates a 20 x 60k
    minusmu2 = csc_matrix((np.log(collapse.toarray())) - minusmu)
    #multiply data by minusmu2 and add to minusmu to obtain Prob(y=1)
    firstprob = (datamodel.multiply(minusmu2[0])).sum(axis=1)
    summedmu = (minusmu[0].sum(axis=1))+ priorsmodel[0,0]
    allprobsY = csc_matrix(firstprob + summedmu)

    for y in range(1,classes):
        probY = (datamodel.multiply(minusmu2[y])).sum(axis=1)
        summedmu = (minusmu[y].sum(axis=1)) + priorsmodel[0,y]
        probY = csc_matrix(probY + summedmu)
        allprobsY = hstack([allprobsY, probY],format="csc") 

    preds = (np.argmax(allprobsY.toarray(),axis=1))
    return preds

def check_error(preds,labelscheck):
    check = ((preds.astype(np.int8))) - (labelscheck.astype(np.int8))
    error = (np.sum(check.astype(np.bool)))/len(preds)
    return error


# In[15]:

#naive bayes classifier
def NaiveBayes (datatrained, datatested, labelstrained, labelstested, classes):
    priors = create_priors(labelstrained,classes)
    collapse = create_mus(datatrained,labelstrained,classes)
    preds = model(collapse,datatested,priors,classes)
    error = check_error(preds,labelstested)
    return error


# In[16]:

#TESTING
#unigram
finaldlabels = np.array(dlabels)
finaltdlabels = np.array(dlabels)

finaldtext = uni_dtext
finaltdtext = uni_dtext

#training data
error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
print(error)



vectorizerT = CountVectorizer(min_df=1, vocabulary=dictuni)
uni_tdtext = vectorizerT.fit_transform(tdtext)

finaltdlabels = np.array(tdlabels)
finaltdtext = uni_tdtext
#test data
error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)


print(error)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



