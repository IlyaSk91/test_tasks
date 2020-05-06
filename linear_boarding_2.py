import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

N=20
# set random points
x,y=np.random.uniform(0,5,size=(N,1)),np.random.uniform(0,5,size=(N,1))
sample=np.concatenate((x,y,np.random.randint(0,2,size=(N,1))),axis=1)
# set initial line
w=np.array([np.random.uniform(),np.random.uniform()])
x1=np.arange(0,6)
y1=w[0]*x1+w[1]

#calculate dist from point to line
dist=[]
for i in range(len(sample)):
    dist.append(np.dot(w,sample[i,:2])/np.linalg.norm(w,ord=2))

res,result=[],[]
accuracy=0
for i in range(len(sample)):
    # perpendicular condition
    k=-1/w[0]
    b=sample[i,1]-k*sample[i,0]
    #calculate pr from current point to line
    A=np.array([ [k,-1],[w[0],-1]])
    B=np.array([-b,-w[1]])
    dec=solve(A,B)
    res.append(dec)
    result.append(sample[i,1]-dec[1])
    # check labels
    if (sample[i,1]-dec[1])>0:
        sign=1
    else:
        sign=0
    if sign==sample[i,2]:
        accuracy=accuracy+1

plt.plot(x1,y1,'k-')
plt.scatter(sample[:,0],sample[:,1],c=sample[:,2],linewidths=10)
plt.xlabel('X')
plt.ylabel('Y')
print('accuracy score','{:.1f}'.format(accuracy/N*100))
