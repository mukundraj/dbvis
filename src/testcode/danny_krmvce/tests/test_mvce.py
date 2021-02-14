#!/usr/bin/env python

# tests for mvce package

import os
import sys
from src.testcode.danny_krmvce.svdd import krmvce

from numpy import dot, outer, zeros, ones, eye, sqrt, diag, where, reshape, isscalar, array, linspace, logical_not, exp, power
from scipy.linalg import svd, norm
from numpy.random import randn
import unittest
import time

import matplotlib.pyplot as plt

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


def test_ellipse(self,K,X,Xtest,gamma,v,kernel,show_plot=False):
	n = K.shape[0]
	start = time.time()
	ellipse = krmvce(K,gamma,v)
	end = time.time()
	print X.shape, "timing: ", end-start
	Ktest = kernel(X,Xtest)
	Kxx = kernel(Xtest,Xtest)
	dists = ellipse.distance(Ktest,Kxx)
	inlier = ellipse.classify(Ktest,Kxx)
	self.assertEqual(False, inlier)
	Ktest = K[:,1]
	Kxx = K[1,1]
	dists = ellipse.distance(Ktest,Kxx)
	inlier = ellipse.classify(Ktest,Kxx)
	self.assertEqual(True, inlier)
	if show_plot:
		plt.figure()
		if X.shape[0] > 2:
			# PCA:
			u,s,v = svd(X)
			X = dot(u[:,0:2].T,X)
		else:
			samples = 100
			x = linspace(X.min(),X.max(),samples)
			y = linspace(X.min(),X.max(),samples)
			for i in range(samples):
				Xtest = zeros([2,samples])
				for j in range(samples):
					Xtest[0,j] = x[i]
					Xtest[1,j] = y[j]
				Ktest = kernel(X,Xtest)
				Kxx = diag(kernel(Xtest,Xtest))
				inliers = ellipse.classify(Ktest,Kxx)
				#print float(sum(inliers))/len(inliers)
				inside = where(inliers)
				outside = where(logical_not(inliers))
				plt.scatter(Xtest[0,inside],Xtest[1,inside],marker='s',color='c')
				plt.scatter(Xtest[0,outside],Xtest[1,outside],marker='s',color='m')
		inlier = ellipse.classify(K,diag(K))
		inside = where(inlier)
		outside = where(logical_not(inlier))
		plt.scatter(X[0,inside],X[1,inside],color='g',marker='x')
		plt.scatter(X[0,outside],X[1,outside],color='g',marker='*')
		minv = min(x.min(), y.min())
		maxv = max(x.max(),y.max())
		plt.xlim(minv,maxv)
		plt.ylim(minv,maxv)
		plt.title("ellipse")
		plt.show()

class TestMvceMethods(unittest.TestCase):

	def test_mvce_tiny(self):
		# small 2D problem:
		d = 2
		n = 4
		X = zeros((d,n))
		X[:,0] = [-10,0]
		X[:,1] = [10,0]
		X[:,2] = [0,-1]
		X[:,3] = [0,1]
		gamma = 1e-1
		v = 1
		Xtest = array([100,100])
		K = linearkernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,linearkernel,False)

	def test_mvce_tiny_kernel(self):
		# small 2D problem:
		d = 2
		n = 4
		X = zeros((d,n))
		X[:,0] = [-10,0]
		X[:,1] = [10,0]
		X[:,2] = [0,-1]
		X[:,3] = [0,1]
		gamma = 1e-1
		v = 1
		sigma = estimate_sigma(X)
		Xtest = array([100,100])
		kernel = lambda x,y: gaussiankernel(x,y,sigma)
		K = kernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,kernel,False)
	
	
	def test_mvce_small(self):
		# small 2D problem:
		d = 2
		n = 20
		cov = eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = dot(cov, randn(d,n)*3)
		gamma = 1e-1
		v = 1
		Xtest = array([100,100])
		K = linearkernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,linearkernel,False)

	def test_mvce_small_kernel(self):
		# small 2D problem:
		d = 2
		n = 20
		cov = eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = dot(cov, randn(d,n)*3)
		gamma = 1e-1
		v = 1
		sigma = estimate_sigma(X)
		Xtest = array([100,100])
		kernel = lambda x,y: gaussiankernel(x,y,sigma)
		K = kernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,kernel,True)
	
	def test_mvce_large(self):
		# larger d-D problem:
		d = 10
		n = 20
		cov = eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = dot(cov, randn(d,n)*3)
		gamma = 1e-1
		v = 1
		Xtest = 100*ones(d)
		K = linearkernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,linearkernel,False)

	def test_mvce_large_kernel(self):
		# larger d-D problem:
		d = 10
		n = 20
		cov = eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = dot(cov, randn(d,n)*3)
		gamma = 1e-1
		v = 1
		sigma = estimate_sigma(X)
		Xtest = 100*ones(d)
		kernel = lambda x,y: gaussiankernel(x,y,sigma)
		K = kernel(X,X)
		test_ellipse(self,K,X,Xtest,gamma,v,kernel,False)
	


if __name__ == '__main__':
    unittest.main()
