"""
Created on 2016-12-05

Class to do Kernel density estimation with cyclic boundary along one direction.

@author: mukundraj

"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from produtils import dprint
import copy
import scipy.spatial.distance as dis

class KerDensity:

    def __init__(self,bw, S):
        """ Builds the kernel density shape

        Args:
            bw: Bandwidth
            X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns:

        """
        self.S = S
        self.bw = bw


    def score_sample(self, X,Y):
        """

        Args:
            X: output of meshgrid
            Y: output of meshgrid
        Returns:

            k_grid: kde estimate (unnormalized) at the data points
        """



        k_grid = np.zeros(np.shape(X))
        maxx = np.amax(X)
        minx = np.amin(X)
        size = maxx - minx
        N = len(self.S)
        den = N*self.bw


        for i in range(N):

            Xl = X - size
            Xr = X + size


            d = ((X-self.S[i][0])**2 + (Y-self.S[i][1])**2)/self.bw
            dl = ((Xl-self.S[i][0])**2 + (Y-self.S[i][1])**2)/self.bw
            dr = ((Xr-self.S[i][0])**2 + (Y-self.S[i][1])**2)/self.bw

            k = np.exp(-d)
            k_l = np.exp(-dl)
            k_r = np.exp(-dr)

            k_grid = k_grid + k + k_l + k_r

        k_grid = k_grid/den

        return k_grid

    def score_sample_colwise(self, X, Y):
        """ Performs the kde columnwise, essentialy 1D kde along Y or second
         member of self.S while keeping Xs independent.

        Args:
            X: output of meshgrid
            Y: output of meshgrid

        Returns:
            k_grid: kde estimate (unnormalized) at the data points

        """
        k_grid = np.zeros(np.shape(X))

        return k_grid


# test code 1
# plt.axis('equal')
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
#
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x + 2;
# # pos[:, :, 0] = x;
# pos[:, :, 1] = y
# dprint(pos.shape)
# rv = multivariate_normal([0.75,0.5])
# CS = plt.contour(x, y, rv.pdf(pos),colors='k')
# plt.clabel(CS, inline=1, fontsize=10)
# plt.show()

# # test code 2
# kd = KerDensity(0.05, [(-0.5,0),(0.9,0.0)])
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# k_grid = kd.score_sample(x,y)
#
# dprint(k_grid.shape)
# plt.axis('equal')
# CS = plt.contour(x, y, k_grid,colors='k')
# plt.clabel(CS, inline=1, fontsize=10)
# plt.show()