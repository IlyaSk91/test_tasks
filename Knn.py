import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv('D:/wine.data')

df=pd.DataFrame(data=None,index=np.arange(0,len(data)+1),columns=np.arange(0,len(data.columns)))
df.iloc[0]=list(data.columns)
df.iloc[1:len(data)]=data.iloc[0:176]
df=df.dropna(axis=0)


y=df[0]
X=df.iloc[:,1:3]
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.3)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train.astype(int))
y_pred=neigh.predict(X_test)
val=y_test.values.astype(int)
print('{:.2f}'.format(accuracy_score(val,(y_pred).astype(int))*100))
matrix=confusion_matrix(val,(y_pred).astype(int))
ax1=plt.subplot(131)
ax1.imshow(matrix)
ax1.set_xlabel('Predict')
ax1.set_ylabel('Actual')
ax1.set_xticks((0,1,2),('1','2','3'))
ax1.set_yticks((0,1,2),('1','2','3'))
for i in range(3):
    for j in range(3):
        ax1.text(i,j,matrix[i,j],ha='center',va='center',color='w')
        
print(classification_report(val,(y_pred).astype(int)))

ax2=plt.subplot(132)


#ax3=plt.subplot(133)
k=0.1
x1,x2=np.meshgrid(np.arange(X[1].astype(float).min()-1,X[1].astype(float).max()+1,k),np.arange(X[2].astype(float).min()-1,X[2].astype(float).max()+1,k))
x=np.array([x1.ravel(),x2.ravel()]).T
Z=neigh.predict(x)

ax2.pcolormesh(x1,x2,Z.reshape(np.shape(x1)),cmap='gist_rainbow_r')
ax2.scatter(X_test[1],X_test[2],c=y_test.values.astype(float))
