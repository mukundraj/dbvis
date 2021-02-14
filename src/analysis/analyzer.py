'''
Created on May 04, 2016

Module to perform depth analysis using various bands for ensemble of vectors.

@author: mukundraj
'''

from scipy.spatial import Delaunay
from itertools import combinations
import numpy as np
import src.analysis.bands as bands
from libs.productivity import dprint
import src.testcode.smallestenclosingcircle as smcir
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import copy
import src.visualization.debug_vis as dbv


class analyzer(object):
    '''
    Class to perform functional band depth analysis on a matrix ensemble.
    '''


    def __init__(self,vecs_list, r, epsilon,vecs_grid=None):
        '''Constructor

        Args:

            vecs_list: a list of vectors
            r    : Number of ensemble members forming the band.
            epsilon: If negative, then generalized band depth.
                    If positive (between 0 and 1), then epsilon fraction must
                     be inside the band to be considered within the band.
            vecs_grid: ??

        Returns:
            None
        '''

        self.r = r
        self.epsilon = epsilon
        self.vecs_list = vecs_list
        self.rect_inband_j = None
        self.ellp_inband_j = None
        self.full_list = None
        if vecs_grid:
            self.full_list = copy.deepcopy(vecs_list)
            self.full_list.extend(vecs_grid)
        else:
            self.full_list = vecs_list
        self.vecs_list = np.array(self.vecs_list)


    def get_combinations(self):
        '''Gets all the combinations in list
        '''
        combs = combinations(range(len(self.vecs_list)),self.r)

        return list(combs)

    def get_rectangle_depth(self):
        """

        Args:
            ensemble:

        Returns:

        """
        if self.full_list is not None: # not present for non kernel point version
            ensize = len(self.full_list)
        else:
            ensize = len(self.vecs_list)

        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.vecs_list[0])

        inband_members = {}
        for comb in combs:
            inband_members[str(list(comb))] = []

        if self.full_list is not None:
            full_list = np.array(self.full_list)
        else:
            full_list=self.vecs_list
        for i in range(ensize):
            inband_count = 0
            for comb in combs:

                subset = full_list[comb,:]

                cur_vec = full_list[i,:]
                max_vec = np.amax(subset, axis = 0)
                min_vec = np.amin(subset, axis = 0)
                top_bounded = cur_vec<=max_vec
                bot_bounded = cur_vec>=min_vec

                inside = top_bounded & bot_bounded

                if self.epsilon == 1:
                    if (np.sum(inside) == vec_len):
                        inband_count = inband_count + 1
                        inband_members[str(list(comb))].append(i)

                elif self.epsilon == -1:
                        inband_count = inband_count + np.sum(inside)/vec_len
                else:
                    exit('Code up the epsilon band depth')

            depths[i] = inband_count/float(len(combs))

        self.rect_inband_j = inband_members
        return depths

    def get_bands_rect_j(self):
        """Returns the band in-band info for each combination.

        Returns:

        """

        return self.rect_inband_j

    def get_bands_ellp_j(self):
        """Returns the band in-band info for each combination with ellipse band.

        Returns:

        """
        return self.ellp_inband_j

    def get_full_list(self):
        """

        Returns:

        """
        return self.full_list

    def get_ellipse_depth(self):
        """

        Returns:

        """
        ensize = len(self.full_list)
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.vecs_list[0])

        inband_members = {}
        for comb in combs:
            inband_members[str(list(comb))] = []

        full_list = np.array(self.full_list)
        for i in range(ensize):
            inband_count = 0

            for comb in combs:

                subset = full_list[comb,:]
                X = full_list[i,:]

                # form the ellipse
                A,C = bands.mvee(subset)

                #Visualization
                #dbv.show_minvol_ellipse(subset,A,C)

                X = np.asmatrix(X)
                C = np.asmatrix(C)
                A = np.asmatrix(A)



                # dprint(np.shape(A),np.shape(C.T),np.shape(X))

                # check if cur_vec is inside
                inout = (X-C) * A * (X-C).T - 1

                # increment inband_count if inside
                if inout <= 0:
                    inband_count += 1
                    inband_members[str(list(comb))].append(i)


            depths[i] = inband_count/float(len(combs))

        self.ellp_inband_j = inband_members

        return depths



    def get_rectangle_depth_grid(self, res):
        """Computes rectangle depth for points on a 2d grid. This is for drawing contours

        Args:
        res: Resolution or number of divisions in the grid.
        Returns:
            grid_depth_info: dict has depths, x_poses, and y_poses
        """
        vec_len = 2
        grid_depth_info = {}
        combs = self.get_combinations()

        x_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
        x_poses = np.around(x_poses,3)
        y_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
        y_poses = np.around(y_poses,3)



        depths = []
        depths_ellipse = []
        depths_sphere = []
        depths_hull = []

        for ypos in y_poses:
            row = []
            row_ellp = []
            row_sphere = []
            row_hull = []
            dprint(ypos)
            for xpos in x_poses:
                # if xpos>0.5:
                #     xpos = 0

                inband_count = 0
                inband_count_ellp = 0
                inband_count_sphere = 0
                inband_count_hull = 0
                for comb in combs:

                    subset = self.vecs_list[comb,:]

                    cur_vec = np.array([xpos,ypos])
                    max_vec = np.amax(subset, axis = 0)
                    min_vec = np.amin(subset, axis = 0)
                    top_bounded = cur_vec<=max_vec
                    bot_bounded = cur_vec>=min_vec

                    inside = top_bounded & bot_bounded

                    if self.epsilon == 1:
                        if (np.sum(inside) == vec_len):
                            inband_count = inband_count + 1
                    elif self.epsilon == -1:
                            inband_count = inband_count + np.sum(inside)/vec_len
                    else:
                        exit('Code up the epsilon band depth')

                    # Now for ellipse
                    subset = self.vecs_list[comb,:]
                    X = cur_vec
                    # form the ellipse
                    A,C = bands.mvee(subset)
                    X = np.asmatrix(X)
                    C = np.asmatrix(C)
                    A = np.asmatrix(A)
                    inout = (X-C) * A * (X-C).T - 1

                    # Now for sphere
                    C = smcir.make_circle(subset)
                    cen = C[:-1]
                    r = C[-1]
                    dis = np.linalg.norm(X-cen)

                    # Now for hull
                    hull = MultiPoint(subset).convex_hull
                    X = np.squeeze(np.array(X))
                    point = Point(X[0],X[1])
                    inout_hull = hull.contains(point)

                    # increment inband_count if inside
                    if inout <= 0:
                        inband_count_ellp += 1
                    if dis <= r:
                        inband_count_sphere += 1
                    if inout_hull:
                        inband_count_hull += 1

                row_hull.append(round(inband_count_hull/float(len(combs)),3))
                row_sphere.append(round(inband_count_sphere/float(len(combs)),3))
                row_ellp.append(round(inband_count_ellp/float(len(combs)),3))
                row.append(round(inband_count/float(len(combs)),3))

            depths.append(row)
            depths_ellipse.append(row_ellp)
            depths_sphere.append(row_sphere)
            depths_hull.append(row_hull)

        depths = [row for row in reversed(depths)]
        depths_ellipse = [row for row in reversed(depths_ellipse)]
        depths_sphere = [row for row in reversed(depths_sphere)]
        depths_hull = [row for row in reversed(depths_hull)]


        grid_depth_info["x_poses"] = [round(x,3) for x in list(x_poses)]
        grid_depth_info["y_poses"] = [round(y,3) for y in list(y_poses)]
        grid_depth_info["depths"] = depths
        grid_depth_info["depths_ellipse"] = depths_ellipse
        grid_depth_info["depths_sphere"] = depths_sphere
        grid_depth_info["depths_hull"] = depths_hull

        return grid_depth_info

    def in_hull(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed

        http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
        """

        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    def get_hull_depth(self):
        """


        """
        ensize = len(self.vecs_list)
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.vecs_list[0])

        self.vecs_list = np.array(self.vecs_list)
        for i in range(ensize):
            inband_count = 0
            for comb in combs:

                subset = self.vecs_list[comb,:]
                X = self.vecs_list[i,:]

                # form the sphere
                hull = MultiPoint(subset).convex_hull
                X = np.squeeze(np.array(X))
                point = Point(X[0],X[1])
                inout_hull = hull.contains(point)
                #dprint(np.shape(A),np.shape(C.T),np.shape(X))

                # increment if inside
                if inout_hull:
                    inband_count += 1

            depths[i] = inband_count/float(len(combs))

        return depths



    def get_sphere_depth(self):
        """

        Args:
            ensemble:

        Returns:

        """
        ensize = len(self.vecs_list)
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.vecs_list[0])

        self.vecs_list = np.array(self.vecs_list)
        for i in range(ensize):
            inband_count = 0
            for comb in combs:

                subset = self.vecs_list[comb,:]
                X = self.vecs_list[i,:]

                # form the sphere
                C = smcir.make_circle(subset)
                cen = C[:-1]
                r = C[-1]
                #dprint(np.shape(A),np.shape(C.T),np.shape(X))

                # check if cur_vec is inside
                dis = np.linalg.norm(X-cen)

                # increment inband_count if inside
                if dis <= r:
                    inband_count += 1


            depths[i] = inband_count/float(len(combs))

        return depths


