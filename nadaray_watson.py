import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

data_train=pd.read_csv('E:/housing_price/train.csv')
data_test=pd.read_csv('E:/housing_price/test.csv')

df=data_train[['YearBuilt','SalePrice']].sort_values(by=['YearBuilt'])
x1=df['YearBuilt'].unique()
y1=df[['YearBuilt','SalePrice']].groupby(by='YearBuilt').mean()
#plt.plot(x1,y1)


x=np.linspace(2,6,100)
y=np.sin(x)+np.random.random(size=(len(x),1))[:,0]


def kernel(z):
    f=(1/(np.sqrt(2*np.pi)))*np.exp(-np.power(z,2)/2)
    return f

#plt.plot(np.arange(1,10,0.1),kernel(np.arange(1,10,0.1)))

h=1.15

def nadaray_watson(x,y):
    est=[]
    for i in range(len(x)):
       a=list()
       for j in range(len(x)):
           a.append(kernel(distance.euclidean(x[i],x[j]/h)))
       w=np.array(a)
       est.append(np.sum(y*w)/np.sum(w))
    return est
 
plt.scatter(x,y,color='k',label='original distribution')
plt.plot(x,nadaray_watson(x,y)*x,label='nadaray-watson')
plt.legend()