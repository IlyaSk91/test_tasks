import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

step=100
x=np.arange(0,step)
data=np.random.uniform(1,20,step)
val=np.random.permutation(np.random.randint(0,step,10))
data[val]=np.nan

print('NaN indexes',[i for i in range(len(data)) if np.isnan(data[i])])

data_without_nan=np.copy(data)
for i in range(len(data_without_nan)):
    if np.isnan(data_without_nan[i]):
        #pass
       np.put(data_without_nan,i,data_without_nan[i-1].mean())

data_drop_nan=data[~np.isnan(data)]
x_drop_nan=np.arange(np.array([data_drop_nan.nonzero()]).min(),np.array([data_drop_nan.nonzero()]).max()+1)
tck = interpolate.InterpolatedUnivariateSpline(x_drop_nan,data_drop_nan)
xnew=np.arange(0,step)
ynew=tck(xnew)
s=90
plt.plot(x[1:s],data[1:s],'*',xnew[1:s],ynew[1:s],x[1:s],data_without_nan[1:s])
plt.legend(['original','spline','interpolation'],fontsize=14)
plt.xlabel('time')
plt.ylabel('sample')

#difference between 2 algorithms results
err=abs(-np.linalg.norm(ynew[1:s],2)+np.linalg.norm(data_without_nan[1:s],2))/np.linalg.norm(data_without_nan[1:s],2)*100
print('difference between 2 algorithms results','{:.2f}'.format(err),'%')