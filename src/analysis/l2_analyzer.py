'''
Created on Feb 24, 2018

Class for computing the l2 depth.

@author: mukundraj
'''

from scipy import linalg
import numpy as np
from produtils import dprint
from scipy.spatial.distance import euclidean as euclidean

class analyzer:

    def __init__(self, members=None,G=None):
        """

        Args:
            members (list of m-d points): Nxm data in euclidean space (N=num of points)

        Returns:

        """

        if members is not None:
            self.X = members
        if G is not None:
            self.G = G
            self.deltas = self.compute_deltas()


    def compute_deltas(self):
        """Computes and stores the delta matrix

        Args:
            xid: The index of 'x' whose depth is being computed.
        Returns:

        """
        m,n = np.shape(self.G)
        assert(m==n)
        deltas = np.zeros((m,n))

        for i in range(m):
            for j in range(n):
                deltas[i,j] = self.G[i,i] + self.G[j,j] - 2*self.G[i,j]

        deltas[deltas<0] = 0
        deltas = np.sqrt(deltas)



        return deltas


    def get_depths(self):
        """returns the Mahalanobis depth in the Euclidean space.

        Returns:

        """
        N = len(self.X)
        dists = np.zeros((N,N))

        # VI = linalg.inv(np.cov(np.array(self.X).T))

        # dprint(mahalanobis(self.X[0],self.X[1],VI))

        for i in range(N):
            for j in range(i):
                dists[i,j] = euclidean(self.X[i],self.X[j])
                dists[j,i] = dists[i,j]

        depths = np.sum(dists, axis=0)
        depths = depths/np.max(depths)
        depths = 1-depths

        return depths


    def get_depths_from_gram(self):
        """

        Returns:
            depths

        """
        N = len(self.X)
        dists = np.zeros((N,N))

        # VI = linalg.inv(np.cov(np.array(self.X).T))

        # dprint(mahalanobis(self.X[0],self.X[1],VI))

        for i in range(N):
            for j in range(i):
                dists[i,j] = self.deltas(self.X[i],self.X[j])
                dists[j,i] = dists[i,j]

        depths = np.sum(dists, axis=0)
        depths = depths/np.max(depths)
        depths = 1-depths

        return depths