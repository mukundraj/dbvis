# test code for testing monotonicity Dec 5, 2016

import numpy as np
from produtils import dprint
from matplotlib import cm
import matplotlib.pyplot as plt
import src.utils.Monotonicity as mono
from sklearn.neighbors import KernelDensity
from matplotlib.legend_handler import HandlerLine2D

def fx(x):

    return 3.66*x - 8*(x**2) + 5.33*(x**3)

res = 100
x = np.linspace(0,1,res)
ya = np.linspace(0,1,res)

y = fx(x)

y = y.reshape(res,1)
ya = ya.reshape(res,1)
dprint(np.shape(ya))

kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(y)
kde2 = np.exp(kde.score_samples(ya))
kde2 = kde2/np.sum(kde2) # normalize for the grid size

g = np.cumsum(kde2)
# g = g/np.sum(kde2)
# g = g/np.sum(y)
dprint(len(g), len(x), len(y))

# plt.figure()
plt.figure(facecolor='white')
line1, = plt.plot(1-x,y,color='b',label="original fn")
# plt.plot(1-kde2,x,lineStyle=':',color='r',label="1-kde/10")
plt.plot(1-2*kde2,x,lineStyle=':',color='r',label="density estimate")
plt.plot(1-x,g,'--',color='g', label='cdf')
plt.plot(1-g,x,color='g', label='monotonized fn')
plt.legend(loc=1,fontsize=18)
plt.show()
