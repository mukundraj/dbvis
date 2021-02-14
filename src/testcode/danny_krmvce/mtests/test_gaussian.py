'''
Simple test file to generate image for an ellipse band in kernel space using gaussian kernel.
'''

from libs.productivity import dprint
from src.testcode.danny_krmvce.svdd import krmvce
from numpy import zeros,array,dot,exp,power
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot
import pylab

# simple example:
d = 2
n = 4
X = zeros((d,n))
X[:,0] = [-10,0]
X[:,1] = [10,5]
X[:,2] = [0,-5]
X[:,3] = [0,5]
dprint(X[0,:])
Xtest = array([[100,100,1],[100,100,1]])
# Xtest = array([100,100])
# parameters
gamma = 1e-1
v = 1

##########################
# gaussian kernel
##########################

def estimate_sigma(X):
    """ estimate the sigma parameter by computing the average minimum distance between points.
    """
    d,n = X.shape
    sigma = 0
    for i in range(n):
        dist = 1e10
        for j in range(n):
            if i == j:
                continue
            dist = min(dist, norm(X[:,i]-X[:,j]))
        sigma += dist
    sigma /= n
    return sigma

def gaussiankernel(x,y,sigma):
    """ compute the Gaussian kernel where k(x,y) = exp( - ||x-y||^2/(2*sigma^2))
    """
    n1 = 1
    if len(x.shape) > 1:
        n1 = x.shape[1]
    n2 = 1
    if len(y.shape) > 1:
        n2 = y.shape[1]
    K = zeros([n1,n2])
    for i in range(K.shape[0]):
        if n1 > 1:
            xx = x[:,i]
        else:
            xx = x
        for j in range(K.shape[1]):
            if n2 > 1:
                yy = y[:,j]
            else:
                yy = y
            K[i,j] = exp( -(norm(xx-yy)**2) / (2*(sigma**2)) )
    return K

sigma = estimate_sigma(X)

K = gaussiankernel(X,X,sigma)
ellipse = krmvce(K,gamma,v)

res = 50

x = np.linspace(-20, 20,res)
y = np.linspace(-20, 20,res)
xx,yy = np.meshgrid(x,y)
xx = xx.flatten()
yy = yy.flatten()

Xtest = array([xx,yy])

Ktest = gaussiankernel(X,Xtest,sigma)
Kxx = gaussiankernel(Xtest,Xtest,sigma)
inlier = ellipse.classify(Ktest,Kxx) # should be False
dprint(inlier)


colors = []
for i in range(len(inlier)):
    if inlier[i]:
        colors.append('r')
    else:
        colors.append('g')

matplotlib.pyplot.scatter(xx,yy,c=colors)
matplotlib.pyplot.scatter(X[0,:],X[1,:],c='b')

matplotlib.pyplot.show()