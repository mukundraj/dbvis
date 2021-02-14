#!/usr/bin/env python

# tests for svdd package

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from src.testcode.danny_krmvce.svdd import svdd as svdd
from src.testcode.danny_krmvce.svdd import esvdd as esvdd
from src.testcode.danny_krmvce.svdd import esvdd_linear as esvdd_linear

import numpy as np
import unittest
import time

import matplotlib.pyplot as plt

def test_sphere(self,K,X,Xtest,C,show_plot=False):
	n = K.shape[0]
	start = time.time()
	sphere = svdd(K,C=1)
	end = time.time()
	print X.shape, "timing: ", end-start
	Ktest = np.dot(X.T,Xtest)
	Kxx = np.dot(Xtest.T,Xtest)
	dists = sphere.distance(Ktest,Kxx)
	inlier = sphere.classify(Ktest,Kxx)
	self.assertEqual(False, inlier)
	Ktest = K[:,1]
	Kxx = K[1,1]
	dists = sphere.distance(Ktest,Kxx)
	inlier = sphere.classify(Ktest,Kxx)
	self.assertEqual(True, inlier)
	center = np.dot(X, sphere.alpha)
	if show_plot:
		if X.shape[0] > 2:
			# PCA:
			u,s,v = np.linalg.svd(X)
			X = np.dot(u[:,0:2].T,X)
		plt.scatter(X[0,:],X[1,:],color='g',marker='x')
		plt.scatter(center[0],center[1],color='r',marker='o')
		plt.plot([center[0,0],center[0,0]+np.sqrt(sphere.radius)],[center[1,0],center[1,0]], 'r--')
		plt.plot([center[0,0]-np.sqrt(sphere.radius),center[0,0]],[center[1,0],center[1,0]], 'r--')
		plt.plot([center[0,0],center[0,0]],[center[1,0],center[1,0]+np.sqrt(sphere.radius)], 'r--')
		plt.plot([center[0,0],center[0,0]],[center[1,0]-np.sqrt(sphere.radius),center[1,0]], 'r--')
		plt.show()

def test_ellipse(self,K,X,Xtest,C,kernel,show_plot=False):
	n = K.shape[0]
	start = time.time()
	ellipse = esvdd(K,C=C)
	end = time.time()
	print X.shape, "timing: ", end-start
	sphere = svdd(K,C=C)
	Ktest = kernel(X,Xtest)
	Kxx = kernel(Xtest,Xtest)
	dists = ellipse.distance(Ktest,Kxx)
	inlier = ellipse.classify(Ktest,Kxx)
	#self.assertEqual(False, inlier)
	Ktest = K[:,1]
	Kxx = K[1,1]
	dists = ellipse.distance(Ktest,Kxx)
	inlier = ellipse.classify(Ktest,Kxx)
	inliersp = sphere.classify(Ktest,Kxx)
	#self.assertEqual(True, inlier)
	center1 = np.dot(X, ellipse.alpha)
	center2 = np.dot(X, ellipse.beta)
	print "centers: ", center1, ", ", center2
	if show_plot:
		plt.figure()
		if X.shape[0] > 2:
			# PCA:
			u,s,v = np.linalg.svd(X)
			X = np.dot(u[:,0:2].T,X)
		else:
			samples = 100
			x = np.linspace(6*np.min(X[0,:]),7*np.max(X[0,:]),samples)
			y = np.linspace(2*np.min(X[1,:]),2*np.max(X[1,:]),samples)
			for i in range(samples):
				Xtest = np.zeros([2,samples])
				for j in range(samples):
					Xtest[0,j] = x[i]
					Xtest[1,j] = y[j]
				Ktest = kernel(X,Xtest)
				Kxx = np.diag(kernel(Xtest,Xtest))
				dists = ellipse.distance(Ktest,Kxx)
				inliers = ellipse.classify(Ktest,Kxx)
				print float(sum(inliers))/len(inliers)
				inside = np.where(inliers)
				outside = np.where(np.logical_not(inliers))
				plt.scatter(Xtest[0,inside],Xtest[1,inside],marker='s',color='c')
				plt.scatter(Xtest[0,outside],Xtest[1,outside],marker='s',color='m')
		plt.scatter(X[0,:],X[1,:],color='g',marker='x')
		plt.scatter(center1[0],center1[1],color='r',marker='o')
		plt.scatter(center2[0],center2[1],color='b',marker='^')
		minv = min(x.min(), y.min())
		maxv = max(x.max(),y.max())
		plt.xlim(minv,maxv)
		plt.ylim(minv,maxv)
		plt.title("ellipse")
		print('now using sphere:')
		plt.figure()
		if X.shape[0] > 2:
			# PCA:
			u,s,v = np.linalg.svd(X)
			X = np.dot(u[:,0:2].T,X)
		else:
			samples = 100
			x = np.linspace(6*np.min(X[0,:]),7*np.max(X[0,:]),samples)
			y = np.linspace(2*np.min(X[1,:]),2*np.max(X[1,:]),samples)
			for i in range(samples):
				Xtest = np.zeros([2,samples])
				for j in range(samples):
					Xtest[0,j] = x[i]
					Xtest[1,j] = y[j]
				Ktest = kernel(X,Xtest)
				Kxx = np.diag(kernel(Xtest,Xtest))
				dists = sphere.distance(Ktest,Kxx)
				inliers = sphere.classify(Ktest,Kxx)
				print float(sum(inliers))/len(inliers)
				inside = np.where(inliers)
				outside = np.where(np.logical_not(inliers))
				plt.scatter(Xtest[0,inside],Xtest[1,inside],marker='s',color='c')
				plt.scatter(Xtest[0,outside],Xtest[1,outside],marker='s',color='m')
		plt.scatter(X[0,:],X[1,:],color='g',marker='x')
		plt.scatter(center1[0],center1[1],color='r',marker='o')
		plt.scatter(center2[0],center2[1],color='b',marker='^')
		minv = min(x.min(), y.min())
		maxv = max(x.max(),y.max())
		plt.xlim(minv,maxv)
		plt.ylim(minv,maxv)
		plt.title("sphere")

		plt.show()

def linearkernel(x,y):
	return np.dot(x.T,y)
def gaussiankernel(x,y,sigma):
	n1 = 1
	if len(x.shape) > 1:
		n1 = x.shape[1]
	n2 = 1
	if len(y.shape) > 1:
		n2 = y.shape[1]
	K = np.zeros([n1,n2])
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
			K[i,j] = np.exp( -np.power(np.linalg.norm(xx-yy),2) / (2*np.power(sigma,2)) )
	return K


def test_ellipse_linear(self,K,X,Xtest,C,kernel,show_plot=False):
	n = K.shape[0]
	start = time.time()
	ellipse = esvdd_linear(X,C=C)
	end = time.time()
	print X.shape, "timing: ", end-start
	sphere = svdd(K,C=C)
	Ktest = kernel(X,Xtest)
	Kxx = kernel(Xtest,Xtest)
	dists = ellipse.distance(Xtest)
	inlier = ellipse.classify(Xtest)
	#self.assertEqual(False, inlier)
	Ktest = K[:,1]
	Kxx = K[1,1]
	dists = ellipse.distance(Xtest)
	inlier = ellipse.classify(Xtest)
	inliersp = sphere.classify(Ktest,Kxx)
	#self.assertEqual(True, inlier)
	center1 = ellipse.C1
	center2 = ellipse.C2
	print "centers: ", center1, ", ", center2
	if show_plot:
		plt.figure()
		if X.shape[0] > 2:
			# PCA:
			u,s,v = np.linalg.svd(X)
			X = np.dot(u[:,0:2].T,X)
		else:
			samples = 100
			x = np.linspace(6*np.min(X[0,:]),7*np.max(X[0,:]),samples)
			y = np.linspace(2*np.min(X[1,:]),2*np.max(X[1,:]),samples)
			for i in range(samples):
				Xtest = np.zeros([2,samples])
				for j in range(samples):
					Xtest[0,j] = x[i]
					Xtest[1,j] = y[j]
				Ktest = kernel(X,Xtest)
				Kxx = np.diag(kernel(Xtest,Xtest))
				inliers = ellipse.classify(Xtest)
				print float(sum(inliers))/len(inliers)
				inside = np.where(inliers)
				outside = np.where(np.logical_not(inliers))
				plt.scatter(Xtest[0,inside],Xtest[1,inside],marker='s',color='c')
				plt.scatter(Xtest[0,outside],Xtest[1,outside],marker='s',color='m')
		inlier = ellipse.classify(X)
		inside = np.where(inlier)
		outside = np.where(np.logical_not(inlier))
		plt.scatter(X[0,inside],X[1,inside],color='g',marker='x')
		plt.scatter(X[0,outside],X[1,outside],color='g',marker='*')
		plt.scatter(center1[0],center1[1],color='r',marker='o')
		plt.scatter(center2[0],center2[1],color='b',marker='^')
		minv = min(x.min(), y.min())
		maxv = max(x.max(),y.max())
		plt.xlim(minv,maxv)
		plt.ylim(minv,maxv)
		plt.title("ellipse")
		#plt.show()
		print('now using sphere:')
		plt.figure()
		if X.shape[0] > 2:
			# PCA:
			u,s,v = np.linalg.svd(X)
			X = np.dot(u[:,0:2].T,X)
		else:
			samples = 100
			x = np.linspace(6*np.min(X[0,:]),7*np.max(X[0,:]),samples)
			y = np.linspace(2*np.min(X[1,:]),2*np.max(X[1,:]),samples)
			for i in range(samples):
				Xtest = np.zeros([2,samples])
				for j in range(samples):
					Xtest[0,j] = x[i]
					Xtest[1,j] = y[j]
				Ktest = kernel(X,Xtest)
				Kxx = np.diag(kernel(Xtest,Xtest))
				dists = sphere.distance(Ktest,Kxx)
				inliers = sphere.classify(Ktest,Kxx)
				print float(sum(inliers))/len(inliers)
				inside = np.where(inliers)
				outside = np.where(np.logical_not(inliers))
				plt.scatter(Xtest[0,inside],Xtest[1,inside],marker='s',color='c')
				plt.scatter(Xtest[0,outside],Xtest[1,outside],marker='s',color='m')
		plt.scatter(X[0,:],X[1,:],color='g',marker='x')
		plt.scatter(center1[0],center1[1],color='r',marker='o')
		plt.scatter(center2[0],center2[1],color='b',marker='^')
		minv = min(x.min(), y.min())
		maxv = max(x.max(),y.max())
		plt.xlim(minv,maxv)
		plt.ylim(minv,maxv)
		plt.title("sphere")
		print "ellipse: "
		print "C1: ", ellipse.C1
		print "C2: ", ellipse.C2
		print "r: ", ellipse.radius

		print "energy: ", ellipse.energy(X)
		ellipse.C1[1] += 5
		ellipse.C2[1] -= 5
		print "modified energy: ", ellipse.energy(X)
		plt.show()



class TestSvddMethods(unittest.TestCase):

	def test_esvdd_linear(self):
		# small 2D problem:
		d = 2
		n = 100
		cov = np.eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = np.dot(cov, np.random.randn(d,n)*3)
		C = 1e4
		Xtest = np.array([100,100])
		K = linearkernel(X,X)
		test_ellipse_linear(self,K,X,Xtest,C,linearkernel,True)
	
	
	def test_svdd(self):
		return None
		# small 2D problem:
		d = 2
		n = 100
		cov = np.eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = np.dot(cov, np.random.randn(d,n)*3)
		K = np.dot(X.T,X)
		C = 1
		Xtest = np.array([100,100])
		test_sphere(self,K,X,Xtest,C,False)
		# larger D problem
		d = 50
		n = 1000
		cov = np.eye(d,d)
		for i in range(d):
			cov[i,i] = np.sqrt(i+1)
		X = np.dot(cov, np.random.randn(d,n)*3)
		K = np.dot(X.T,X)
		C = 1
		Xtest = np.ones(d)*1e8
		test_sphere(self,K,X,Xtest,C,False)

	def test_esvdd(self):
		return None
		# small 2D problem:
		d = 2
		n = 100
		cov = np.eye(d,d)
		cov[0,0] = 2
		cov[1,1] = 10
		X = np.dot(cov, np.random.randn(d,n)*3)
		C = 1
		Xtest = np.array([100,100])
		if False:
			K = linearkernel(X,X)
			test_ellipse(self,K,X,Xtest,C,linearkernel,True)
		else:
			sigma = 0
			for i in range(n):
				dists = 1e10*np.ones(n)
				for j in range(n):
					if i != j:
						dists[j] = np.linalg.norm(X[:,j]-X[:,i])
				sigma += dists.min()
			sigma /= n
			print "sigma = ", sigma
			kernel = lambda x,y: gaussiankernel(x,y,sigma)
			K = kernel(X,X)
			test_ellipse(self,K,X,Xtest,C,kernel,True)
	

if __name__ == '__main__':
    unittest.main()
