'''
Created on Oct 10, 2016

Module to perform half space depth analysis.

@author: mukundraj
'''
import copy
from itertools import combinations,permutations
import ctypes

from produtils import dprint
import numpy as np
import src.datarelated.processing.depths as dp


class analyzer(object):
    '''
    Class to perform half space depth analysis using only dot products.
    '''

    def __init__(self, members, grid_members, kernel=None, G=None):
        '''Constructor

        Args:

            members: A list of members
            grid_members: A list of members to play the role of grid points
            kernel: The kernel to use
            G: Precomputed gram matrix

        Returns:
            None
        '''

        self.members = members
        self.kernel = kernel
        self.grid_members = grid_members
        if grid_members:
            self.all_members = copy.deepcopy(members)
            self.all_members.extend(grid_members)
        else:
            self.all_members = members
        if G is not None:
            self.G=G

        # self.vecs_list = np.array(self.vecs_list)


    def get_depths(self):
        '''
        Computes and returns the kernel half space depth.

        Returns:
            depths (list of double): depth values for all the members.
            inside_list: List of lists of indices of members constituting the min halfspace.

        '''

        inside_list = [] # A list of lists of indices of members

        # compute all the direction vectors
        dirs = []
        combs = permutations(range(len(self.members)),2)
        for c in combs:
            dirs.append(self.members[c[0]]-self.members[c[1]])


        depths = []
        S = [m for m in self.members]
        for q in self.all_members:

            minval = np.Inf
            min_halfspace_members = None
            for a in dirs:

                ctr = 0
                halfspace_members = []
                for i,p in enumerate(S):
                    if self.kernel(a, p) >= self.kernel(a,q):
                        halfspace_members.append(i)
                        ctr = ctr+1


                if ctr<minval:
                    minval=ctr
                    min_halfspace_members = halfspace_members


            depths.append(minval)
            inside_list.append(min_halfspace_members)

            inside_list = inside_list[:len(self.members)] # Don't need the info for the gridpoints

        depths = np.array(depths)
        return depths, self.all_members, inside_list



    def get_depths_gram(self):
        """Computes and returns the kernel half space depth
        using the gram matrix given instead of the kernel function.

        Returns:
            depths (list of double): depth values for all the members.
            inside_list: List of lists of indices of members constituting the min halfspace.

        """

        N_members = self.G.shape[0]
        depths = []
        inside_list = [] # A list of lists of indices of members

        # compute all the direction vectors
        dirs = []
        perms = permutations(range(len(self.members)),2)
        dirs = [p for p in perms]
        dprint(len(dirs))

        for q in range(N_members):
            dprint('q',q)
            minval = np.Inf
            min_halfspace_members = None

            for a in dirs:
                ctr = 0
                halfspace_members = []

                for i in range(N_members):
                    if self.G[a[0],i] - self.G[a[1],i] >= self.G[a[0],q] - self.G[a[1],q]:
                        halfspace_members.append(i)
                        ctr = ctr+1

                if ctr<minval:
                    minval=ctr
                    min_halfspace_members = halfspace_members

            depths.append(minval)
            inside_list.append(min_halfspace_members)

            inside_list = inside_list[:len(self.members)] # Don't need the info for the gridpoints

        depths = np.array(depths)

        return depths, inside_list

    def get_depths_extern(self):
        """Computes and returns the depths using Dyckerhoff C++ code.

        Returns:
            depths (list of double): depth values for all the members.
        #     g++ -shared HD.cpp -std=c++11 -O2 -o HD.so

        """

        _sum = ctypes.CDLL('/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/src/extern/cpp/HD.so')
        _sum.our_function.argtypes = (ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.c_double))
        _sum.our_function.restype = ctypes.c_double

        points2d = np.array(self.members)
        N,d = np.shape(points2d)
        array_type = ctypes.c_double * (N * d)
        depths = []
        # loop and center each point
        for i in range(N):

            c_points2d = copy.deepcopy(points2d)
            for j in range(d):
                c_points2d[:,j] -= c_points2d[i,j]

            # if i==13:
            #     dprint(c_points2d)

            x = []
            for k in range(d):
                x = x+list(c_points2d[:,k])
            depth_i = _sum.our_function(ctypes.c_int(N), ctypes.c_int(d), array_type(*x))
            depths.append(depth_i)





        # depths = dp.histeq(np.array(depths))
        return np.array(depths)
