'''
Created on Feb 21, 2017

Class for computing the mahalanobis depth.

@author: mukundraj
'''

from scipy import linalg
import numpy as np
from produtils import dprint
from scipy.spatial.distance import mahalanobis as mahalanobis

class MahaDepth:

    def __init__(self, members):
        """

        Args:
            members (list of m-d points): Nxm data in euclidean space (N=num of points)

        Returns:

        """

        self.X = members


    def get_depths(self):
        """returns the Mahalanobis depth in the Euclidean space.

        Returns:

        """
        N = len(self.X)
        dists = np.zeros((N,N))

        VI = linalg.inv(np.cov(np.array(self.X).T))

        # dprint(mahalanobis(self.X[0],self.X[1],VI))

        for i in range(N):
            for j in range(i):
                dists[i,j] = mahalanobis(self.X[i],self.X[j],VI)
                dists[j,i] = dists[i,j]

        depths = np.sum(dists, axis=0)
        depths = depths/np.max(depths)
        depths = 1-depths

        return depths
