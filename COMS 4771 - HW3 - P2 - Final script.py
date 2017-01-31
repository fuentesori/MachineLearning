
# coding: utf-8

# In[1]:

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


# In[2]:

start_time = time.time()
#runing with 200,000 first rows of training data
samplerows = 1000
testrows = 1000
validations =5
data = pd.read_csv('reviews_tr.csv', nrows=samplerows, iterator=True)
tdata = pd.read_csv('reviews_te.csv', nrows=testrows, iterator=True)

#split data into text and labels
dlabels = data['label']
dtext = data['text']
tdlabels = tdata['label']
tdtext = tdata['text']


# In[3]:

#set up representations
#unigram
vectorizer = CountVectorizer(min_df=1)
#binary vectorizer for bayes
vectorizerbinary = CountVectorizer(min_df=1, binary=True)
#term frequency rep
tfvect = TfidfVectorizer(smooth_idf=False)
#bigram represetation
vectorizer2 = CountVectorizer(min_df=1, ngram_range=(2,2))
#drop common words
vectorizer3 = CountVectorizer(min_df=1, stop_words=('the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
))


# In[4]:

#create representations and dictionaries to fit test data
#binary unigram representation for bayes (ie. 1s or 0s)
unibi_dtext = vectorizerbinary.fit_transform(dtext)
dictunibi = vectorizerbinary.get_feature_names()

#unigram representation
uni_dtext = vectorizer.fit_transform(dtext)
dictuni = vectorizer.get_feature_names()

#Term frequency-inverse document frequency tf-idf
tf_dtext = tfvect.fit_transform(dtext)
dicttf = tfvect.get_feature_names()

#adjust to correct log
tf_dtext = tf_dtext/(np.log(10))

#bigram representation
bi_dtext = vectorizer2.fit_transform(dtext)
dictbi = vectorizer2.get_feature_names()

#unigram removing 20 most common english words
uni2_dtext = vectorizer3.fit_transform(dtext)
dictuni2 = vectorizer3.get_feature_names()


# In[5]:

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


# In[6]:

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


# In[7]:

#naive bayes classifier
def NaiveBayes (datatrained, datatested, labelstrained, labelstested, classes):
    priors = create_priors(labelstrained,classes)
    collapse = create_mus(datatrained,labelstrained,classes)
    preds = model(collapse,datatested,priors,classes)
    error = check_error(preds,labelstested)
    return error


# In[8]:

kf = KFold(samplerows, n_folds=validations)
errormatrix=[]
for train_index, test_index in kf:
    
    crossdlabels = np.array(dlabels[train_index])
    crosstdlabels = np.array(dlabels[test_index])
    #run unigram that is binary on bayes
    crossdtext = unibi_dtext[train_index]
    crosstdtext = unibi_dtext[test_index]
    
    error = NaiveBayes(crossdtext, crosstdtext, crossdlabels, crosstdlabels,2)
    errormatrix.append(error)
    #run unigram on perceptron
    crossdtext = uni_dtext[train_index]
    crosstdtext = uni_dtext[test_index]
     
    error = Perceptron(crossdtext, crosstdtext, crossdlabels, crosstdlabels)
    errormatrix.append(error)
    #run bigram on perceprton
    crossdtext = bi_dtext[train_index]
    crosstdtext = bi_dtext[test_index]
     
    error = Perceptron(crossdtext, crosstdtext, crossdlabels, crosstdlabels)
    errormatrix.append(error)
    #run tf on perceptron
    crossdtext = tf_dtext[train_index]
    crosstdtext = tf_dtext[test_index]
     
    error = Perceptron(crossdtext, crosstdtext, crossdlabels, crosstdlabels)
    errormatrix.append(error)
    #run ex 20 common names on perceptron
    crossdtext = uni2_dtext[train_index]
    crosstdtext = uni2_dtext[test_index]
     
    error = Perceptron(crossdtext, crosstdtext, crossdlabels, crosstdlabels)
    errormatrix.append(error)

errormatrix = np.reshape(errormatrix,(validations,5))
print(errormatrix)
averror = np.mean(errormatrix,axis=0)
print(averror)
method = np.argmin(averror)
print(method)
print("--- %s seconds ---" % (time.time() - start_time))


# In[9]:

#test on test data
finaldlabels = np.array(dlabels)
finaltdlabels = np.array(dlabels)

if method==0:
    #binary vectorizer for bayes
    finaldtext = unibi_dtext    
    finaltdtext = unibi_dtext
    
    #training
    error = NaiveBayes(finaldtext, finaltdtext, finaldlabels, finaltdlabels,2)
    print("training error:",error)   
    
    #test
    vectorizerbinaryT = CountVectorizer(min_df=1, binary=True, vocabulary=dictunibi)
    unibi_tdtext = vectorizerbinaryT.fit_transform(tdtext)
    
    finaltdtext = unibi_tdtext
    finaltdlabels = np.array(tdlabels)
    
    error = NaiveBayes(finaldtext, finaltdtext, finaldlabels, finaltdlabels,2)
    print("test error:",error)
    
elif method==1:
    #unigram
    finaldtext = uni_dtext
    finaltdtext = uni_dtext
    
    #training
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("training error:",error)
    
    #test
    vectorizerT = CountVectorizer(min_df=1, vocabulary=dictuni)
    uni_tdtext = vectorizerT.fit_transform(tdtext)
    finaltdtext = uni_tdtext
    finaltdlabels = np.array(tdlabels)
    
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("test error:",error)
    
elif method ==2:
    #bigram
    finaldtext = bi_dtext
    finaltdtext = bi_dtext
    
    #training
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("training error:",error)
    
    #test
    vectorizer2T = CountVectorizer(min_df=1, vocabulary=dictbi)
    bi_tdtext = vectorizer2T.fit_transform(tdtext)
    finaltdtext = bi_tdtext
    finaltdlabels = np.array(tdlabels)
    
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("test error:",error)
    
elif method ==3:
    #tf method
    finaldtext = tf_dtext
    finaltdtext = tf_dtext
    
    #training
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)    
    print("training error:",error)
    
    #test
    tfvectT = TfidfVectorizer(smooth_idf=False, vocabulary=dicttf)
    tf_tdtext = tfvectT.fit_transform(tdtext)
    finaltdtext = tf_tdtext
    finaltdlabels = np.array(tdlabels)
    
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("test error:",error)
    
elif method ==4:
    #common
    finaldtext = uni2_dtext
    finaltdtext = uni2_dtext
    
    #training
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("training error:",error)
    
    #test
    vectorizer3T = CountVectorizer(min_df=1, stop_words=('the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'), vocabulary=dictuni2)
    uni2_tdtext = vectorizer3T.fit_transform(tdtext)
    finaltdtext = uni2_tdtext
    finaltdlabels = np.array(tdlabels)
    
    error = Perceptron(finaldtext, finaltdtext, finaldlabels, finaltdlabels)
    print("test error:",error)
else:
    error =0

print("Method:",method,"test error:",error)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



