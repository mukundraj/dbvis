'''
Created on Jun 16, 2016

Module with kernel definition functions.

@author: mukundraj
'''
import numpy as np
from scipy.linalg import norm
from produtils import dprint

class kernels(object):


    def __init__(self, X=None, kernelname=None, G=None):




        if X==None:
            assert(kernelname==None)
            self.G=G
        else:
            self.sigma = self.estimate_sigma(X)
             # self.sigma = 1
            self.kernels = {}
            self.kernels['linearkernel'] = self.linearkernel
            self.kernels['gaussiankernel'] = self.gaussiankernel
            self.kernel = self.kernels[kernelname]
            self.X = X

    def linearkernel(self, x, y):
        """ compute the linear kernel
        """
        return np.dot(x.T,y)


    ##########################
    # gaussian kernel
    ##########################

    def estimate_sigma(self, X):
        """ estimate the sigma parameter by computing the average minimum distance between points.
        """
        X = np.array(X)
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

    def gaussiankernel(self, x,y):
        """ compute the Gaussian kernel where k(x,y) = exp( - ||x-y||^2/(2*sigma^2))
        """
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
                K[i,j] = np.exp( -(norm(xx-yy)**2) / (2*(self.sigma**2)) )
        return K



    def get_gram_matrix_euclidean(self):
        """Uses the data and the kernel to compute and return a gram matrix.
        Works with data that are points in euclidean space.

        Returns:

        """
        m,n = np.shape(self.X)

        G = np.zeros((m,m))

        for i in range(m):
            for j in range(i+1):
                G[i,j] = self.kernel(self.X[i],self.X[j])
                G[j,i] = G[i,j]

        return G

    def get_distance_from_gram(self):
        """Computes the distance matrix using the gram matrix.

        Returns:

        """

        m,n = np.shape(self.X)
        D = np.zeros((m,m))
        G = self.get_gram_matrix_euclidean()

        for i in range(m):
            for j in range(i+1):
                d2 = G[i,i] - 2*G[i,j] + G[j,j]
                D[i,j] = np.sqrt(d2)
                D[j,i] = D[i,j]

        np.fill_diagonal(D, 0)

        return D

    def get_distance_from_precomputed_gram(self):
        """This version when the gram matrix G has been already provided.

        Returns:

        """
        m,n = np.shape(self.G)
        D = np.zeros((m,m))
        G = self.G
        dprint('###ALERT####ALERT#### check for neg distances')
        for i in range(m):
            for j in range(i+1):
                d2 = G[i,i] - 2*G[i,j] + G[j,j]

                if d2<=0:
                    d2=0.0001
                D[i,j] = np.sqrt(d2)
                D[j,i] = D[i,j]

        np.fill_diagonal(D, 0)

        dprint(np.count_nonzero(D==0))
        dprint(np.shape(D))


        return D