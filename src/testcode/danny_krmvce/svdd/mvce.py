#!/usr/bin/env python

from numpy import dot, outer, zeros, ones, eye, sqrt, diag, max, min, where, reshape, isscalar
from scipy.linalg import svd, norm, solve, pinv #,chol,ldl
import cvxpy
import numpy as np
import numpy.linalg as la

from libs.productivity import dprint


class KernelEllipse:
	def __init__(self,alpha,gamma,v,K):
		alpha = where(alpha.ravel() < 0, zeros(len(alpha)).ravel(), alpha.ravel()).T
		alpha = reshape(alpha, [len(alpha),])
		self.alpha = alpha
		self.A = sqrt(alpha)
		self.gamma = gamma
		self.v = v
		AKA = dot(dot(diag(self.A),K),diag(self.A))
		self.V,self.L,U = svd(AKA)
		self.Lsqrt = sqrt(self.L)
		self.Linvsqrt = diag(pinv(diag(sqrt(self.L))))
		mahal = self.distance(K,diag(K))
		self.radius = max(mahal)

	def distance(self, Ktest, Kxx):
		"""compute mahalanobis distance of test points
		"""
		n = Ktest.shape[0]
		if len(Ktest.shape) == 1:
			Ktest = reshape(Ktest,(n,1))
		m = Ktest.shape[1]
		if isscalar(Kxx):
			Kxx = reshape(Kxx,(1,1))

		if len(Kxx.shape)>1:
			Kxx = np.diagonal(Kxx)
		dists = zeros(m)

		for i in range(m):
			KA = dot( Ktest[:,i].T, diag(self.A) )
			KAV = dot( KA, self.V )
			KAVL = dot( KAV, diag(self.Lsqrt) )
			LVtAK = dot( diag(self.Linvsqrt), KAV.T )
			LinvLVtAK = solve( diag(self.L)+eye(n)*self.gamma, LVtAK )
			dists[i] = (1/self.gamma) * ( Kxx[i] -  dot( KAVL, LinvLVtAK ) )
		return dists

	def classify(self, Ktest, Kxx):
		"""classify a test vector
		params:
		Ktest - n x m, where n is train size, m is the test size 
		Kxx - m elements, corresponding to k(x,x) for each test element
		"""
		dists = self.distance(Ktest,Kxx)
		inliers = zeros(len(dists),bool)
		for i in range(len(dists)):
			inliers[i] = dists[i] <= self.radius
		return inliers

def solve_krmvce(K,gamma,v):
	"""solve for minimum volume covering ellipse using cvxpy
	"""
	n = K.shape[0]
	#C = cholesky(K)
	#L,D = ldl(K) # from special branch of numpy on github
	#D = diag(D)
	#D = where(D < 0, zeros([len(D),]), D)
	L,D,L2 = svd(K)
	Dsqrt = diag(sqrt(D))
	C = dot(L,Dsqrt)
	alpha = cvxpy.Variable(n,1)
	c1 = alpha >= 0
	#c3 = v*n*alpha <= 1
	if v == 0:
		c3 = alpha <= 1
	else:
		c3 = v*n*alpha <= 1

	c2 = sum(alpha) == 1
	M = cvxpy.Variable(n,n)
	M = alpha[0]*outer(C[:,0],C[:,0].T)
	for i in range(1,n):
		M += alpha[i]*outer(C[:,i],C[:,i].T)
	c = [c1,c2,c3]
	obj = cvxpy.Minimize(-cvxpy.log_det(M + eye(n)*gamma))
	prob = cvxpy.Problem(obj, c)
	minval = prob.solve()
	return alpha.value

def krmvce(K,gamma=1,v=0):
	"""computes the kernel regularized minimum volume coverint ellipse of a dataset.
	params:
	K - gram matrix of data
	gamma  - regularization parameter, default = 1.
	v - penalty coefficient for slack (soft margin)
	"""
	alpha = solve_krmvce(K,gamma,v)
	return KernelEllipse(alpha,gamma,v,K)


################# -- IGNORE code below here, needs to be updated --- ########################

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
		dists = zeros(m)
		for i in range(m):
			if m > 1:
				xx = Xtest[:,i]
			else:
				xx = Xtest
			dists[i] = norm(xx - self.C1)  +  norm(xx - self.C2)
		return dists

	def classify(self, Xtest):
		"""classify a test vector
		params:
		Xtest - d x n where n is train size, d is the data dimension
		"""
		dists = self.distance(Xtest)
		inliers = zeros(len(dists),bool)
		for i in range(len(dists)):
			inliers[i] = dists[i] <= self.radius
		return inliers
	def energy(self, X):
		d = X.shape[0]
		n = X.shape[1]
		energy = 0
		for i in range(n):
			energy = max(energy, norm( X[:,i] - self.C1 ) + norm( X[:,i] - self.C2 ) )
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


class GramEllipse:
	def __init__(self,M,alpha,X):
		alpha = where(alpha.ravel() < 0, zeros(len(alpha)).ravel(), alpha.ravel()).T
		alpha = reshape(alpha, [len(alpha),])
		self.alpha = alpha
		self.M = M
		mahal = self.distance(X)
		self.radius = max(mahal)
	def distance(self, X):
		"""compute mahalanobis distance of test points
		"""
		if len(X.shape) == 1:
			X = X.reshape((X.shape[0],1))
		d,m = X.shape
		dists = zeros(m)
		for i in range(m):
			dists[i] = dot(X[:,i].T, solve(self.M, X[:,i]))

		return dists
	def classify(self, X):
		"""classify a test vector
		params:
		X - d x m, where d is dimension, m is the test size
		"""
		dists = self.distance(X)
		inliers = zeros(len(dists),bool)
		for i in range(len(dists)):
			inliers[i] = dists[i] <= self.radius
		return inliers


def solve_mvce_linear_gram(X, gamma=0):
	"""solve for minimum volume covering ellipse with Gram form using cvxpy
	"""
	d,n = X.shape
	alpha = cvxpy.Variable(n,1)
	c1 = alpha >= 0
	c2 = sum(alpha) == 1
	M = cvxpy.Variable(n,n)
	M = alpha[0]*outer(X[:,0],X[:,0].T)
	for i in range(1,n):
		M += alpha[i]*outer(X[:,i],X[:,i].T)
	M += gamma*eye(d)
	c = [c1,c2]
	obj = cvxpy.Minimize(-cvxpy.log_det(M))
	prob = cvxpy.Problem(obj, c)
	minval = prob.solve()
	#minval = prob.solve(solver=cvxpy.SCS, eps=1e-10)
	return M.value,alpha.value

def mvce(X,gamma=0,use_gram=True):
	"""computes the extended support vector data description (esvdd) of a dataset.
	params:
	X - matrix of data
	gamma - regularization parameter
	"""
	if use_gram:
		M,alpha = solve_mvce_linear_gram(X,gamma)
		return GramEllipse(M,alpha,X)
	else:
		raise Exception('Not working, use gram formulation instead.')
		M,radius = solve_mvce_linear(X)
		return Ellipse(M,radius)


def mvee(points, gamma = 0, tol = 1e-4):
	"""
	Find the minimum volume ellipse.
	gamma is a regularization parameter to handle degenerate and high dimensional cases, gamma = 0 means no regularization.
	Return A, c where the equation for the ellipse given in "center form" is
	(x-c).T * A * (x-c) = 1
	"""
	points = np.asmatrix(points).T
	N, d = points.shape
	Q = np.column_stack((points, np.ones(N))).T
	err = tol+1.0
	u = np.ones(N)/N
	Gamma = gamma * eye(d+1)
	Gamma[-1,-1] = 0
	while err > tol:
			# assert u.sum() == 1 # invariant
			X = Q * np.diag(u) * Q.T + Gamma
			#M = np.diag(Q.T * la.inv(X) * Q)
			M = np.diag(Q.T * la.solve(X,Q))
			jdx = np.argmax(M)
			step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
			new_u = (1-step_size)*u
			new_u[jdx] += step_size
			err = la.norm(new_u-u)
			u = new_u
	c = u*points
	A = la.inv(points.T*np.diag(u)*points + gamma*eye(d) - c.T*c)/d
	return Ellipse(np.asarray(A), np.squeeze(np.asarray(c)), tol)