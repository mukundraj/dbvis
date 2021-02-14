#!/usr/bin/env python

import numpy as np
import cvxpy

class KernelEllipse:
	def __init__(self,alpha,beta,radius,K,C):
		self.alpha = alpha
		self.alphaKalpha = np.dot(alpha.T,np.dot(K,alpha))[0]
		self.beta = beta
		self.betaKbeta = np.dot(beta.T,np.dot(K,beta))[0]
		self.radius = radius
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
			dists[i] = Kxx[i] - 2*np.dot(np.reshape(self.alpha,(n,)),Ktest[:,i]) + self.alphaKalpha + \
			           Kxx[i] - 2*np.dot(np.reshape(self.beta,(n,)),Ktest[:,i]) + self.betaKbeta 
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
		
def kernel_norm(K, i, weight):
	#return cvxpy.sqrt( K[i,i] - 2*K[i,:]*weight + cvxpy.quad_form(weight, K) )
	return np.sqrt( K[i,i] - 2*np.dot(K[i,:],weight) + cvxpy.quad_form(weight, K) )

def solve_mee(K,C):
	"""solve for minimum enclosing ellipse using cvxpy
	"""
	n = K.shape[0]
	Kdiag = np.diag(K)
	alpha = cvxpy.Variable(n)
	beta = cvxpy.Variable(n)
	eta = cvxpy.Variable(n)
	r = cvxpy.Variable()
	c1 = (eta >= 0)
	c2 = (alpha >= 0)
	c3 = (beta >= 0)
	c = [c1,c2,c3]
	# sqrt => concave
	# quad_form on spd => convex
	# quad_form on snd => concave
	for i in range(n):
		#c.append( kernel_norm(K,i,alpha) + kernel_norm(K,i,beta) <= r + eta[i] )
		#c.append( cvxpy.sqrt2( K[i,i] - 2*K[i,:]*alpha + cvxpy.quad_form(alpha,K) ) + cvxpy.sqrt2( K[i,i] - 2*K[i,:]*beta + cvxpy.quad_form(beta,K) ) <= r + eta[i] )
		c.append(  K[i,i] - 2*K[i,:]*alpha + cvxpy.quad_form(alpha,K) + K[i,i] - 2*K[i,:]*beta + cvxpy.quad_form(beta,K)  <= r + eta[i] )
	obj = cvxpy.Minimize(r + C*sum(eta))
	prob = cvxpy.Problem(obj, c)
	minval = prob.solve()
	return alpha.value,beta.value,r.value

def esvdd(K,C=1):
	"""computes the extended support vector data description (esvdd) of a dataset.
	params:
	K - gram matrix of data
	C - regularization parameter, default = 1.
	"""
	alpha,beta,radius = solve_mee(K,C)
	return KernelEllipse(alpha,beta,radius,K,C)


class Ellipse:
	def __init__(self,C1,C2,radius):
		self.C1 = C1
		self.C2 = C2
		self.radius = radius
	def distance(self, Xtest):
		"""compute distance from center points
		"""
		if len(Xtest.shape) > 1:
			d = Xtest.shape[0]
			m = Xtest.shape[1]
		else:
			d = len(Xtest)
			m = 1
		dists = np.zeros(m)
		for i in range(m):
			if m > 1:
				xx = Xtest[:,i]
			else:
				xx = Xtest
			dists[i] = np.linalg.norm(xx - self.C1)  +  np.linalg.norm(xx - self.C2)
		return dists
	def classify(self, Xtest):
		"""classify a test vector
		params:
		Xtest - d x n where n is train size, d is the data dimension
		"""
		dists = self.distance(Xtest)
		inliers = np.zeros(len(dists),bool)
		for i in range(len(dists)):
			inliers[i] = dists[i] <= self.radius
		return inliers
	def energy(self, X):
		d = X.shape[0]
		n = X.shape[1]
		energy = 0
		for i in range(n):
			energy = max(energy, np.linalg.norm( X[:,i] - self.C1 ) + np.linalg.norm( X[:,i] - self.C2 ) )
		return energy
		
def solve_mee_linear(X,C):
	"""solve for minimum enclosing ellipse using cvxpy for linear problem
	"""
	d = X.shape[0]
	n = X.shape[1]
	C1 = cvxpy.Variable(d)
	C2 = cvxpy.Variable(d)
	r = cvxpy.Variable()
	#eta = cvxpy.Variable(n)
	#c1 = (eta >= 0)
	#c = [c1]
	c = []
	for i in range(n):
		#c.append( cvxpy.norm(X[:,i]-C1) + cvxpy.norm(X[:,i]-C2) <= r + eta[i] )
		c.append( (cvxpy.norm(X[:,i]-C1) + cvxpy.norm(X[:,i]-C2)) <= r )
	#obj = cvxpy.Minimize(r + C*sum(eta))
	obj = cvxpy.Minimize(r)
	prob = cvxpy.Problem(obj, c)
	minval = prob.solve()
	print "prob min val = ", minval
	return C1.value,C2.value,r.value

def esvdd_linear(X,C=1):
	"""computes the extended support vector data description (esvdd) of a dataset.
	params:
	X - matrix of data
	C - regularization parameter, default = 1.
	"""
	C1,C2,radius = solve_mee_linear(X,C)
	return Ellipse(C1,C2,radius)
