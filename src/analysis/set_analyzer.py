'''
Created on Nov 10, 2017

Module to perform set band depth analysis.

@author: mukundraj
'''


import src.analysis.kcat.kernels.functions as kfs
import src.analysis.kcat.datasets as ds
import numpy as np
import src.analysis.halfspace_analyzer as hsa
import src.analysis.kernels as kers
import src.datarelated.processing.dim_reduction as dr
from produtils import dprint
import src.visualization.rmds as rmds
import src.datarelated.readwrite.datacsvs as rw
from itertools import combinations






class analyzer:

    def __init__(self, X,r):
        """

        Args:
            X: A 2D array. Collection of sets, where each member has been encoded by a
            numerical attribute.
            r: Number of members forming the band.

        Returns:

        """

        self.X = X
        self.r = r


    def get_combinations(self):
        '''Gets all the combinations in list
        '''

        mat_shape = np.shape(self.X)
        combs = combinations(range(mat_shape[0]),self.r)

        return list(combs)



    def get_set_depth(self):
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

            set_list = [set(self.X[i,:]) for i in comb]

            union = set.union(*set_list)
            inter = set.intersection(*set_list)

            # loop through each variable

            for i in range(m):

                cur_set = set(self.X[i,:])


                # do a band test, and add to all_depths

                if cur_set.issubset(union) and inter.issubset(cur_set):
                    depths[i] += 1

        depths = depths/len(combs)

        return depths




#
# X = np.array([[1,4,7], [1,4,7], [2,4, 7], [2,5,7], [3,6,8]])
#
# ana = analyzer(X,2)
#
# ana.get_set_depth()