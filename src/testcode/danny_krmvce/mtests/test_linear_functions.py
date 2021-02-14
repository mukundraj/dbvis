'''
Simple test file to generate image for an ellipse band in kernel space using linear kernel.
This file is for high dimensional objects as found in functions.
'''
from libs.productivity import dprint
from src.testcode.danny_krmvce.svdd import krmvce
from numpy import zeros,array,dot,exp,power
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot
import pylab


# simple example:
d = 1000
n = 4
X = zeros((d,n))
X[:,0] = np.random.rand(d)*5
X[:,1] = np.random.rand(d)*(-5)
X[:,2] = np.random.rand(d)*5
X[:,3] = np.random.rand(d)*(-5)
dprint(X[:,0])
Xtest = array([[100,100,1],[100,100,1]])
# Xtest = array([100,100])
# parameters
gamma = 1e-1
v = 1

##########################
# linear kernel
##########################

def linearkernel(x,y):
    """ compute the linear kernel
    """
    return dot(x.T,y)


K = linearkernel(X,X) # ie dot(X.T,X)
ellipse = krmvce(K,gamma,v)

dprint("done",len(ellipse.alpha), ellipse.alpha)

# Ktest = linearkernel(X,Xtest)
#
# Kxx = linearkernel(Xtest,Xtest)
#
# res = 100
#
# x = np.linspace(-20, 20,res)
# y = np.linspace(-20, 20,res)
# xx,yy = np.meshgrid(x,y)
# xx = xx.flatten()
# yy = yy.flatten()
#
# Xtest = array([xx,yy])
#
# Ktest = linearkernel(X,Xtest)
# Kxx = linearkernel(Xtest,Xtest)
# inlier = ellipse.classify(Ktest,Kxx) # should be False
# dprint(inlier)
#
# colors = []
# for i in range(len(inlier)):
#     if inlier[i]:
#         colors.append('r')
#     else:
#         colors.append('g')
#
# matplotlib.pyplot.scatter(xx,yy,c=colors)
# matplotlib.pyplot.scatter(X[0,:],X[1,:],c='b')
#
# matplotlib.pyplot.show()