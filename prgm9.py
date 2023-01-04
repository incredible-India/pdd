# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:17:09 2022

@author: user
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as npl
def kernel(point,xmat,k):
    m,n=npl.shape(xmat)
    weights=npl.mat(npl.eye(m))
    for j in range(m):
        diff=point-xmat[j]
        weights[j,j]=npl.exp(diff*diff.T/(-2.0*k**2))
    return weights
def localweight(point,xmat,ymat,k):
    wei=kernel(point,xmat,k)
    w=(xmat.T*(wei*xmat)).I*(xmat.T*(wei*ymat.T))
    return w
def localweightregression(xmat,ymat,k):
    row,col=npl.shape(xmat)
    ypred=npl.zeros(row)
    for i in range(row):
        ypred[i]=xmat[i]*localweight(xmat[i],xmat,ymat,k)
    return ypred
data=pd.read_csv(r'data10.csv')
bill=npl.array(data.total_bill)
tip=npl.array(data.tip)
mbill=npl.mat(bill)
mtip=npl.mat(tip)
mbillMatCol=npl.shape(mbill)[1]
onesArray=npl.mat(npl.ones(mbillMatCol))
xmat=npl.hstack((onesArray.T,mbill.T))
ypred=localweightregression(xmat,mtip,2)
SortIndex=xmat[:,1].argsort(0)
xsort=xmat[SortIndex][:,0]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='blue')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('tip')
plt.show();



