#!/usr/bin/env python

import numpy as np
import cvxpy

class KernelSphere:
	def __init__(self,alpha,K,C):
		self.alpha = alpha
		self.alphaKalpha = np.dot(alpha.T,np.dot(K,alpha))[0]
		# to compute radius, need any element for which alpha < C
		#ind = np.where( (self.alpha < C) & (self.alpha > 1e-5) )[0]
		#self.radius = K[ind[0],ind[0]] - 2*np.dot(K[:,ind[0]],alpha) + self.alphaKalpha
		self.radius = (np.dot(np.diag(K),alpha) - self.alphaKalpha)
		self.radius = self.radius[0,0]
	def distance(self, Ktest, Kxx):
		"""compute distance from center of points
		"""
		n = Ktest.shape[0]
		if len(Ktest.shape) == 1:
			Ktest = np.reshape(Ktest,(n,1))
		m = Ktest.shape[1]
		if np.isscalar(Kxx):
			Kxx = np.reshape(Kxx,(1,1))
		dists = np.zeros(m)
		for i in range(m):
			dists[i] = Kxx[i] - 2*np.dot(np.reshape(self.alpha,(n,)),Ktest[:,i]) + self.alphaKalpha
		return dists
	def classify(self, Ktest, Kxx):
		"""classify a test vector
		params:
		Ktest - n x m, where n is train size, m is the test size 
		Kxx - m elements, corresponding to k(x,x) for each test element
		"""
		dists = self.distance(Ktest,Kxx)
		inliers = np.zeros(len(dists),bool)
		for i in range(len(dists)):
			inliers[i] = dists[i] <= self.radius
		return inliers


def solve_meb(K,C):
	"""solve for minimum enclosing ball using cvxpy
	"""
	n = K.shape[0]
	Kdiag = np.diag(K)
	alpha = cvxpy.Variable(n)
	c1 = (alpha >= 0)
	c2 = (alpha <= C)
	c3 = (sum(alpha) == 1)
	obj = cvxpy.Maximize(Kdiag.T*alpha - cvxpy.quad_form(alpha, K))
	prob = cvxpy.Problem(obj, [c1,c2,c3])
	maxval = prob.solve()
	return alpha.value

def svdd(K,C=1):
	"""computes the support vector data description (svdd) of a dataset.
	params:
	K - gram matrix of data
	C - regularization parameter, default = 1.
	"""
	alpha = solve_meb(K,C)
	return KernelSphere(alpha,K,C)
