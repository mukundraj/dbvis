'''
Created on Jun 16, 2016

Module to perform depth analysis using kernel ellipse band depth.

@author: mukundraj
'''

import copy
from itertools import combinations

import numpy as np

from libs.productivity import dprint
# from src.testcode.danny_krmvce.svdd import krmvce,mvce

from mivel.mvee import mvee
from numpy import zeros,array,dot,exp,power
from scipy.linalg import norm
import src.visualization.debug_vis as dbv
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
from  scipy.spatial.distance import euclidean

class kernel_analyzer(object):
    '''Class to perform kernel depth analysis

    '''


    def __init__(self, pts_list, gridpts, r, gamma, v, kernel, show_3d_bands):
        '''Constructor

        '''

        self.pts_list = pts_list
        self.gridpts = gridpts
        self.r = r
        self.v = v
        self.kernel = kernel
        self.Kxx = None # Gram matrix of test points
        self.combs = None
        self.dist_mat = self.compute_dist_mat()
        self.show_3d_bands = show_3d_bands

        self.gamma = gamma
        if self.gamma<0:
            self.gamma_v1 = self.estimate_gamma()
            dprint('estimated gamma', self.gamma_v1)
        else:
            dprint('gamma', self.gamma)

        # For visualization
        self.rect_details = []

        # # for generating pca points
        # self.full_list = copy.deepcopy(pts_list)
        # self.full_list.extend(gridpts)
    #
    # def get_full_list(self):
    #     """
    #
    #     Returns:
    #
    #     """
    #     return self.full_list


    def get_center(self, points):
        """

        Args:
            points: A numpy array of points

        Returns:
            cens: a list with the mean in each dimension.

        """
        cens = []
        dims = np.shape(points)[1]
        length = len(points)
        for i in range(dims):
            cur_sum = np.sum(points[:,i])
            cens.append(cur_sum/length)

        return cens


    def get_point2d_gram_matrix(self, points2d):
        """ returns the gram matrix using kernel

        Args:
            points2d: list of ensemble members

        Returns:
            K: Gram matrix

        """
        N = len(points2d)
        K = np.zeros(shape=(N,N))

        for i in range(N):
            for j in range(N):
                K[i,j] = self.kernel(points2d[i],points2d[j])
                #K[j,i] = kernel(points2d[j],points2d[i])


        return K

    def get_smallest_n(self, arr, k):
        """

        Args:
            arr: input array
            k: smallest k

        Returns:
            smallest: Returns the smallest two members from the input array
        """


        smallest_args = np.argpartition(arr, k)[:k]
        smallest = arr[smallest_args]

        return smallest

    def estimate_gamma(self):
        """Selects gamma as the median of the global k nearest neighbors.

        Args:
            points: The original points.

        Returns:
            gamma: the estimated value of gamma

        """
        smallest = []

        N = len(self.pts_list)
        dist_mat_inf = np.empty((N,N))
        dist_mat_inf[:,:] = np.Inf
        pairs = combinations(range(N), 2)

        for pair in pairs:

            i1 = pair[0]
            i2 = pair[1]
            dist_mat_inf[i1,i2] = np.sqrt(self.kernel(self.pts_list[i1],self.pts_list[i2]))

        # dprint(dist_mat_inf[0,:])
        N = len(self.pts_list)
        for i in range(N):
            smallest.extend(self.get_smallest_n(dist_mat_inf[i,:],2))

        gamma = np.median(smallest)
        dprint(gamma)

        return gamma

    def compute_dist_mat(self):
        """

        Returns:

        """

        N = len(self.pts_list)
        dist_mat = np.zeros((N,N))
        pairs = combinations(range(N), 2)

        for pair in pairs:

            i1 = pair[0]
            i2 = pair[1]
            # dist_mat[i1,i2] = np.sqrt(self.kernel(self.pts_list[i1],self.pts_list[i2]))
            # todo: figure out how this needs to be modified when we start working with nontrivial kernels
            dist_mat[i1,i2] = euclidean(self.pts_list[i1],self.pts_list[i2])
            dist_mat[i2,i1] = dist_mat[i1,i2]

        return dist_mat

    def estimate_gamma_v2(self, pointids):
        """Selects gamma subject to an upper bound which is the maximum distance between
        two points in the set of points.

        Uses the ids to get the maximum pairwise distance from a precomputed distance matrix.

        Args:
            pointids: The original points.

        Returns:
            gamma: the estimated value of gamma

        """
        gamma = 0

        combs = combinations(pointids, 2)
        dists = [self.dist_mat[x[0],x[1]] for x in combs]
        gamma = np.max(dists)
        return gamma



    def get_combinations(self):
        '''Gets all the combinations in list
        '''
        combs = combinations(range(len(self.pts_list)), self.r)
        self.combs = list(combs)
        return self.combs

    def check_gamma_validity(self, rmvce_min, d_max):
        """Tests if gamma value is safe to guarantee the uniqueness of the mvee.

        Args:
            rmvce_min (float): Length of the minor axis of the gamma-regularized mvce
            d_max (float): Max distance between any two points in points enclosed by the ellipse.
        Returns:
            validity (boolean): True or False based on the validity of gamma.n
        """

        if rmvce_min <= d_max:
            return True
        else:
            return False


    def get_ellipses(self):
        """

        Args:
            K: The gram matrix of the data cloud.

        Returns:
            ellipses_list: A list of ellipse descriptions.
            Ktest_list: A list containing the Ktest matrices for the corresponding
                ellipse.

        """
        ellipses_list = []
        Ktest_list = []
        Kxx_list = []
        cur_Kxx_proj_list = []

        pts_array = np.array(self.pts_list)
        # dprint(np.shape(pts_array))
        # exit(0)
        combs = self.get_combinations()

        Kxx_proj = copy.deepcopy(self.pts_list)
        Kxx_proj.extend(self.gridpts)
        N_Kxx = len(Kxx_proj)

        self.Kxx = np.zeros(shape=(N_Kxx,N_Kxx))

        dims = np.shape(Kxx_proj)[1]

        dprint(np.shape(Kxx_proj),dims)
        for i in range(len(combs)):
            cur_pts = copy.deepcopy(pts_array[combs[i],:])
            # center points, get cur_cen
            cur_cen = self.get_center(cur_pts)

            for j in range(dims):
                cur_pts[:,j] = cur_pts[:,j] - cur_cen[j]

            cur_K = self.get_point2d_gram_matrix(cur_pts)
            # generate cur ellipse
            # ellipse = krmvce(cur_K,self.gamma,self.v)
            # ellipse = mvce.mvce(cur_K, self.gamma, use_gram=True)
            # ellipse = mvce.mvce(np.transpose(cur_pts), self.gamma, use_gram=True)

            gamma_final = self.gamma
            gamma_v2 = self.estimate_gamma_v2(combs[i])
            if self.gamma<0:
                gamma_final = min(self.gamma_v1,gamma_v2)
                dprint('gamma v2', gamma_v2,gamma_final)

            ellipse = mvee(np.transpose(cur_pts), gamma_final)
            ellipses_list.append(ellipse)

            # dprint("ellipse done", float(i)/len(combs))

            ## Eigen value and axes length stuff.
            w = linalg.eigh(ellipse.M,eigvals_only=True)
            try:
                axeslenths = [2/math.sqrt(x) for x in w]
            except:
                dprint(x)

            dprint('ellipse done', float(i)/len(combs), min(axeslenths), gamma_v2)
            # assert self.check_gamma_validity(rmvce_min=min(axeslenths),d_max=gamma_v2),\
            #     'Gamma value could be too high'




            #Visualization
            if self.show_3d_bands:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                dbv.show_minvol_ellipse(cur_pts,ellipse.M,ellipse.c,ax)
                # dbv.show_3d_rectange(ellipse.c, (2,2,2), self.ax)
                dprint('el_cn', ellipse.c, self.rect_details[i]['center'])
                dbv.show_3d_rectange(self.rect_details[i]['center'], self.rect_details[i]['size'], ax)

                angles = np.linspace(0,360,21)[:-1] # A list of 20 angles between 0 and 360

                # create an animated gif (20ms between frames)
                dbv.rotanimate(ax, angles,'movie.gif',delay=20)
                dprint("done")
                plt.show()

            # center the test points based on cur_cen
            cur_Kxx_proj = copy.deepcopy(np.array(Kxx_proj))

            for j in range(dims):
                cur_Kxx_proj[:,j] = cur_Kxx_proj[:,j] - cur_cen[j]

            cur_Kxx_proj_list.append(cur_Kxx_proj)

            cur_Kxx = np.zeros(shape=(N_Kxx,N_Kxx))
            for i2 in range(N_Kxx):
                for j2 in range(N_Kxx):
                    cur_Kxx[i2,j2] = self.kernel(cur_Kxx_proj[i2],cur_Kxx_proj[j2])

            Kxx_list.append(cur_Kxx)


            # generate cur Ktest
            N_cur = np.shape(cur_K)[0]
            Ktest = np.zeros(shape=(N_cur,N_Kxx))

            for i_r in range(len(combs[i])):
                for i_j in range(N_Kxx):
                        Ktest[i_r,i_j] = self.kernel(cur_pts[i_r,:],cur_Kxx_proj[i_j,:])



            Ktest_list.append(Ktest)




        # cur_Kxx_proj_list- list of centered points centered similar to current band points
        return ellipses_list, Ktest_list, Kxx_proj, Kxx_list, cur_Kxx_proj_list


    def get_depths(self):
        """Get depth values

        Returns:
            depths:
            Kxx_proj: Just the points extended with the grid points.
            inside_list: List of lists of indices of members falling inside band

        """
        inside_list = []
        ellipses_list, Ktest_list, Kxx_proj, Kxx_list, cur_Kxx_proj_list = self.get_ellipses()
        status_mat = np.zeros(shape=(len(ellipses_list),np.shape(self.Kxx)[0]))

        for i,ellipse in enumerate(ellipses_list):

            # inliers = ellipse.classify(Ktest_list[i], Kxx_list[i])
            inliers = ellipse.classify(np.transpose(cur_Kxx_proj_list[i]))
            status_mat[i,:] = inliers
            inside = np.where(inliers==True)[0]

            inside_list.append(inside)

        depths = np.mean(status_mat, axis=0)
        return depths, Kxx_proj, inside_list

    def get_depths_rect(self, epsilon):
        """Gets rect depth values

        Returns:
            depths:
            Kxx_proj: Just the points extended with the grid points.
            inside: List of lists of indices of members falling inside band

        """


        Kxx_proj = copy.deepcopy(self.pts_list)
        Kxx_proj.extend(self.gridpts)
        N_Kxx = len(Kxx_proj)
        all_pts = np.array(Kxx_proj)
        pts_array = np.array(self.pts_list)

        ensize = N_Kxx
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.pts_list[0])

        inside_list = []
        inband_members = {}
        for comb in combs:
            inband_members[str(list(comb))] = []

        # Todo : Optimize following loops by swapping the inner and outer loop order
        for i in range(ensize):
            inband_count = 0
            for comb in combs:

                subset = pts_array[comb,:]

                cur_vec = all_pts[i,:]
                max_vec = np.amax(subset, axis = 0)
                min_vec = np.amin(subset, axis = 0)
                top_bounded = cur_vec<=max_vec
                bot_bounded = cur_vec>=min_vec

                inside = top_bounded & bot_bounded

                if epsilon == 1:
                    if (np.sum(inside) == vec_len):
                        inband_count = inband_count + 1
                        inband_members[str(list(comb))].append(i)

                elif epsilon == -1:
                        inband_count = inband_count + np.sum(inside)/vec_len
                else:
                    exit('Code up the epsilon band depth')

            depths[i] = inband_count/float(len(combs))

        for comb in combs:
            inside_list.append(inband_members[str(list(comb))])

        # Preparing for the rectangle visualization.

        if self.show_3d_bands:
            dims = np.shape(pts_array)[1]
            for comb in combs:
                    subset = pts_array[comb,:]
                    cur_cen = self.get_center(subset)
                    for j in range(dims):
                        subset[:,j] = subset[:,j] - cur_cen[j]
                    max_vec = np.amax(subset, axis=0)
                    min_vec = np.amin(subset, axis=0)
                    center = (max_vec+min_vec)/2
                    size = abs(max_vec - min_vec);

                    self.rect_details.append({'center':center,'size':size})

        return depths, Kxx_proj, inside_list