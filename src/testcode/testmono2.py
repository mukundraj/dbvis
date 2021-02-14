# test code for testing monotonicity Dec 5, 2016
# for non 0-1 domain

import numpy as np
from produtils import dprint
from matplotlib import cm
import matplotlib.pyplot as plt
import src.utils.Monotonicity as mono
from sklearn.neighbors import KernelDensity
from matplotlib.legend_handler import HandlerLine2D
from src.utils.Scaler import Scaler

def fx(x):

    return 3.66*(x+2) - 8*((x+2)**2) + 5.33*((x+2)**3)

res = 100
x = np.linspace(-2,-0.5,res)
y = fx(x)

x_scaler = Scaler(x)
y_scaler = Scaler(y)

x = x_scaler.to_unit(x)
y = y_scaler.to_unit(y)

miny = np.amin(y)
maxy = np.amax(y)
ya = np.linspace(miny,maxy,res)

y = y.reshape(res,1)
ya = ya.reshape(res,1)
dprint(np.shape(ya))
dprint(np.shape(y))
exit(0)

kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(y)
kde2 = np.exp(kde.score_samples(ya))
kde2 = kde2/np.sum(kde2) # normalize for the grid size

g = np.cumsum(kde2)

dprint(g)
# g = g/np.sum(kde2)
# g = g/np.sum(y)

x1 = x_scaler.fr_unit(x)
y = y_scaler.fr_unit(y)


plt.figure()
plt.plot(x1,y,color='b',label="Original fn")

kde2 = y_scaler.fr_unit(kde2)
plt.plot(1-kde2,ya,lineStyle=':',color='r',label="1-kde/10")

g1 = y_scaler.fr_unit(g)
plt.plot(x1,g1,'--',color='g', label='cdf')

g2 = x_scaler.fr_unit(g)
x2 = y_scaler.fr_unit(x)
plt.plot(g2,x2,color='g', label='inv cdf')

plt.legend(loc=2)
plt.show()
