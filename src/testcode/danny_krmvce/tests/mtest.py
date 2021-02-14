from src.testcode.danny_krmvce.svdd import krmvce
import numpy as np
from numpy import dot, outer, zeros, ones, eye, sqrt, diag, where, reshape, isscalar, array, linspace, logical_not, exp, power


import cvxpy

def linearkernel(x,y):
	return dot(x.T,y)
def estimate_sigma(X):
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
			K[i,j] = exp( -power(norm(xx-yy),2) / (2*power(sigma,2)) )
	return K

# simple example:
d = 2
n = 4
X = np.zeros((d,n))
X[:,0] = [-10,0]
X[:,1] = [10,0]
X[:,2] = [0,-1]
X[:,3] = [0,1]
Xtest = np.array([100,100])



# parameters
gamma = 1e-1
v = 1

##########################
# linear kernel
##########################

K = linearkernel(X,X) # ie dot(X.T,X)
ellipse = krmvce(K,gamma,v)

Ktest = linearkernel(X,Xtest)
kernel = lambda x,y: gaussiankernel(x,y,sigma)
Kxx = kernel(Xtest,Xtest)
inlier = ellipse.classify(Ktest,Kxx) # should be False

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True

##########################
# gaussian kernel
##########################

K = gaussiankernel(X,X) # ie K[i,j] = exp(-dot(X[:,i].T,X[:,j])/(2*sigma*sigma))
ellipse = krmvce(K,gamma,v)

Ktest = gaussiankernel(X,Xtest)
Kxx = kernel(Xtest,Xtest)
inlier = ellipse.classify(Ktest,Kxx) # should be False

Ktest = K[:,1]
Kxx = K[1,1]
inlier = ellipse.classify(Ktest,Kxx) # should be True