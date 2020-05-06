import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

#load dataset
data=pd.read_csv('D:/classification/crx.data',names=np.arange(16))
df=data[data!='?'].dropna(axis=0)

cat_columns=[0,3,4,5,6,8,9,11,12,15]
#encode category features
df[cat_columns]=df[cat_columns].astype('category').apply(lambda x:x.cat.codes)
X=df.iloc[:,0:15]
y=df.iloc[:,15]
#different test size samples
test_size=[0.15,0.25,0.65,0.95]

ax=np.arange(1,5)

for t,axs in zip(test_size,ax):
    #split dataset between test sample and train sample
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=t)
    #use LR model and predict values
    clf=LogisticRegression().fit(X_train,y_train)
    y_score=clf.predict(X_test)
    #plot ROC-curve with different test size samples
    fpr,tpr,thresholds=roc_curve(y_test,y_score,pos_label=None)
    #plot curves
    plt.subplot(2,2,axs)
    plt.plot(fpr,tpr,'k-',Linewidth=3)
    plt.grid()
    plt.title('ROC curve for test split equal %.2f'%t,Fontsize=12)
    plt.xlabel('False positive')
    plt.ylabel('True positive')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.95,
wspace=0.95)