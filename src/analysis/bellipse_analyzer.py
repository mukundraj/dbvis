'''
Created on Feb 02, 2018

Module to perform bounding ellipse depth analyzer (different from min vol ellipse).

@author: mukundraj
'''
import copy
from itertools import combinations,permutations
import ctypes

from produtils import dprint
import numpy as np
import src.datarelated.processing.depths as dp
from scipy.spatial.distance import euclidean


class analyzer(object):
    '''
    Class to bounding ellipse depth analysis.
    '''

    def __init__(self, X=None, G=None, eps=1.1):
        '''Constructor

        Args:

            X: list of points

        Returns:
            None
        '''
        self.eps=eps
        if X is not None:
            self.X = X
        if G is not None:
            self.G = G
            self.deltas = self.compute_deltas()


        # self.vecs_list = np.array(self.vecs_list)

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

    def get_depths_from_gram(self):
        """

        Returns:
            depths

        """

        m,n = np.shape(self.G)

        depths = np.zeros(m)

        # get combinations

        combs = self.get_combinations()


        # loop through combinations

        for ind, comb in enumerate(combs):

            dprint(float(ind)/len(combs))

            # get sets

            set_list = comb

            a = comb[0]
            b = comb[1]


            #1.1 for thesis pic
            d = 1.5*self.deltas[a,b]

            # loop through each variable

            for i in range(m):

                x = i

                d_xa = self.deltas[x,a]
                d_xb = self.deltas[x,b]


                # do a band test, and add to all_depths
                # dprint(d_xa+d_xb,d)
                if d_xa + d_xb < d:
                    depths[i] += 1


        dprint (depths)
        depths = depths/len(combs)

        return depths


    def get_combinations(self):
        '''Gets all the combinations in list
        '''

        mat_shape = np.shape(self.X)
        combs = combinations(range(mat_shape[0]), 2)

        return list(combs)



    def get_bellipse_depth(self):
        """

        Returns:
            depths: a 1D array of depth values

        """

        m,n = np.shape(self.X)

        depths = np.zeros(m)

        # get combinations

        combs = self.get_combinations()


        # loop through combinations

        for ind, comb in enumerate(combs):

            dprint(float(ind)/len(combs))

            # get sets

            set_list = [self.X[i] for i in comb]

            a = set_list[0]
            b = set_list[1]
            d = self.eps*euclidean(a,b)
            #
            # union = set.union(*set_list)
            # inter = set.intersection(*set_list)
            #
            # loop through each variable

            for i in range(m):

                x = self.X[i]


                # do a band test, and add to all_depths

                if euclidean(x,a) + euclidean(x,b) < d:
                    depths[i] += 1

        depths = depths/len(combs)

        return depths