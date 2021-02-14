'''
Simple test file to generate image for an ellipse band in kernel space using linear kernel.
'''
from libs.productivity import dprint
from src.testcode.danny_krmvce.svdd import krmvce
from numpy import zeros,array,dot,exp,power
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot
import pylab

def centeroidnp(arr):
    """

    Args:
        arr:

    Returns:
    References:
        http://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi
    """
    length = arr.shape[1]
    dprint(length)
    sum_x = np.sum(arr[0,:])
    sum_y = np.sum(arr[1,:])
    dprint(arr,sum_x,sum_y)
    return sum_x/length, sum_y/length

# simple example:
d = 2
n = 3
X = zeros((d,n))
# X[:,0] = [-10,0]
# X[:,1] = [10,5]
# X[:,2] = [0,-5]
# X[:,3] = [0,5]
X[:,0] = [0.275, 0.53]
X[:,1] = [0.306, 0.304]
X[:,2] = [0.112, 0.25]
X[:,0] = [0.849, 0.179]
X[:,1] = [0.054, 0.362]
X[:,2] = [0.275, 0.53]

cen = centeroidnp(X)
dprint(cen)
X[0,:] = X[0,:] - cen[0]
X[1,:] = X[1,:] - cen[1]


Xtest = array([[100,100,1],[100,100,1]])
# Xtest = array([100,100])
# parameters
gamma = 1e-1
v = 0

##########################
# linear kernel
##########################

def linearkernel(x,y):
    """ compute the linear kernel
    """
    return dot(x.T,y)


K = linearkernel(X,X) # ie dot(X.T,X)
ellipse = krmvce(K,gamma,v)


# Ktest = linearkernel(X,Xtest)
# Kxx = linearkernel(Xtest,Xtest)

res = 15

x = np.linspace(0, 1,res)
y = np.linspace(0, 1,res)
xx,yy = np.meshgrid(x,y)
xx = xx.flatten()
yy = yy.flatten()

Xtest = array([xx,yy])
Xtest[0] = Xtest[0] - cen[0]
Xtest[1] = Xtest[1] - cen[1]


Ktest = linearkernel(X,Xtest)
Kxx = linearkernel(Xtest,Xtest)
inlier = ellipse.classify(Ktest,Kxx) # should be False

colors = []
for i in range(len(inlier)):
    if inlier[i]:
        colors.append('r')
    else:
        colors.append('g')

matplotlib.pyplot.scatter(xx,yy,c=colors)

X[0,:] = X[0,:] + cen[0]
X[1,:] = X[1,:] + cen[1]
matplotlib.pyplot.scatter(X[0,:],X[1,:],c='b')

matplotlib.pyplot.show()