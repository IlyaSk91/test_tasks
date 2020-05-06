import numpy as np
import matplotlib.pyplot as plt

k=50
x=np.linspace(1,10,k)[:, np.newaxis]
y=2*x+3*np.random.random(size=(k,1))
x /= np.max(x)


def grad(w,x,y):
    y_est=w*x
    err=y-y_est
    gradient=-1/len(x)*2*x*err
    return gradient,-1/len(x)*np.power(err,2)

w=np.random.random()
alpha=0.5
tolerance=1e-2

iter=1
while True:
    gradient,error=grad(w,x,y)
    w_new=w-alpha*gradient
    
    
    if np.sum(abs(w_new-w))<tolerance:
        print('converged')
        break
    
    if iter%100==0:
        print(iter,error)
        
    iter=iter+1
    w=w_new
    
p1,=plt.plot(x,y)
p2,=plt.plot(x,w*x)
plt.grid()
plt.legend([p1,p2],['original','reconstruct'])
