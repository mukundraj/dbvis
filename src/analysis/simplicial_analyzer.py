'''
Created on Nov 24, 2017

Module to perform simplicial band depth analysis for arbitrary dimensions.

@author: mukundraj
'''

import numpy as np
from itertools import combinations
from produtils import dprint
from numpy.linalg import svd

import scipy
from scipy import linalg, matrix

class analyzer:

    def __init__(self,X,r):
        """

        Args:
            X: A 2d arrary of points. n x d
            r: points forming the band. must satisfy r > d

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



    def null(self,A, eps=1e-12):
        u, s, vh = scipy.linalg.svd(A)
        padding = max(0,np.shape(A)[1]-np.shape(s)[0])
        null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
        null_space = scipy.compress(null_mask, vh, axis=0)
        return scipy.transpose(null_space)

    def get_simplicial_depth(self):
        """

        Returns:
            depths: a 1D array of depth values

        """

        m,d = np.shape(self.X)

        X = np.array(self.X)

        depths = np.zeros(m)

        # get combinations

        combs = self.get_combinations()

        lenc = float(len(combs))
        # loop through bands

        for ind, comb in enumerate(combs):
            dprint(ind/lenc)

            comb_set = set(comb)
            comb_sign = np.ones(m)


            # loop through faces
            for i in range(len(comb)):
                cur_face_inds = list(comb_set.difference(set([comb[i]])))

                # dprint(X[list(all_inds),:],i)
                M = X[cur_face_inds[:-1],:] - X[cur_face_inds[-1],:]
                # dprint(M)


                N = np.squeeze(self.null(matrix(M)))
                # Nu = np.linalg.norm(nullsp)


                d = np.inner(N, X[cur_face_inds[-1]])

                # get projection of i th point on normal vector
                face_point_sign = np.sign(np.inner(N,X[comb[i],:]) - d)


                # loop through points
                for j in range(m):

                    if j==comb[i]:
                        comb_sign[j] = 0

                    test_point_sign = np.sign(np.inner(N,X[j,:]) - d)

                    # dprint(face_point_sign,test_point_sign)
                    # containment check for test point. if in any subband, point in band
                    if face_point_sign != test_point_sign:

                        comb_sign[j] = 0



            # if comb_sign[19] == 1:
            #     dprint(comb)
                # dprint(comb_sign)


            # dprint(comb_sign)
            depths = depths + comb_sign




        return depths


