'''
Created on Jan 22, 2016

Class to perform the depth constrained MDS for multimodal data with known classes

@author: mukundraj
'''

from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import numpy as np
from libs.productivity import dprint
import copy
import scipy.spatial.distance as distance
import src.utils.interpolation.positions as ipos
import src.analysis.halfspace_analyzer as hsa
import src.datarelated.processing.misc as misc
import src.utils.MonotonicityColwise as monocol
from src.utils.interpolation.Tps import Tps
from scipy.interpolate import griddata
from matplotlib import cm
import csv
import json
from sklearn.metrics.pairwise import linear_kernel
import src.analysis.spatial_analyzer as spa
from PIL import Image
import math
from datarelated.processing.depths import get_median_and_bands

class dc_mds_multimodal(object):
    """

    """

    def __init__(self, depths1, depths2, N_i, M_i,res,bw, max_step,class_boun):
        """

        Args:
            depths1:
            detphs2:
            N_i:
            M_i:
            res:
            bw:
            max_step:
            num_old:
            num_old_spline:

        Returns:

        """

        self.depths1 = depths1
        self.depths2 = depths2

        self.depths1_max = np.amax(self.depths1)
        self.depths1_min = np.amin(self.depths1)
        self.depths2_max = np.amax(self.depths2)
        self.depths2_min = np.amin(self.depths2)

        self.depths1_range = self.depths1_max - self.depths1_min
        self.depths2_range = self.depths2_max - self.depths2_min
        # self.median1_ind = np.argmax(self.depths1)
        # self.median2_ind = np.argmax(self.depths2)+len(self.depths1)
        # dprint('median2_ind',self.median2_ind)

        self.N_i = N_i
        self.M_i = M_i
        self.res = res
        self.bw = bw
        self.max_step = max_step
        self.twice_grid_side = float(2)/res
        # self.gd = None # container to store the gradient to avoid recomputation of gradient.

        self.old_ZIcm_a = None
        self.old_ZIcm_b = None

        # sorted_inds1 = sorted(range(self.depths1.shape[0]), key=lambda k: self.depths1[k])
        # self.ori_median_ind1 = sorted_inds1[-1]
        #
        # sorted_inds2 = sorted(range(self.depths2.shape[0]), key=lambda k: self.depths2[k])
        # self.ori_median_ind2 = sorted_inds2[-1]
        # dprint('ori_median_ind1',self.ori_median_ind1)
        # dprint('ori_median_ind2',self.ori_median_ind2)



        self.adaptive_enabled = False
        self.running_avg_enabled = False

        self.class_boun = class_boun


    def depths_scaler(self, c_depths, orig_depths_range, orig_depths_min):
        """Scales the current depths to have the same range as the input depths.

        Args:
            c_depths: Current depths.

        Returns:

        """
        c_depths_min = np.amin(c_depths)
        c_depths_max = np.amax(c_depths)
        # dprint('maxmin', c_depths_max,c_depths_min)
        ratio = orig_depths_range/(c_depths_max - c_depths_min)

        c_depths = c_depths - c_depths_min
        c_depths = c_depths*ratio + orig_depths_min
        return c_depths

    def list_to_dist_mat(self, pos):
        """

        Args:
            pos (list): list of positions

        Returns:

        """
        n = len(pos)
        dist_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                d = distance.euclidean(pos[i],pos[j])
                if d<=0:
                    d=0.0001
                dist_mat[i,j]  = d
                dist_mat[j,i] = dist_mat[i,j]

        return dist_mat

    def get_depth_energy(self, depths, ZIcm, XI2, YI2, cur_pos, depth_factor):
        """Get the curre

        Args:
            ZIcm (2d array):
            cur_ids (list):

        Returns:
            depth_energy (float):

        """
        cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res, XI2, YI2)
        depths_curpos = np.zeros(len(cur_ids))
        # distance.euclidean(cur_pos[],pos[j])
        for i in range(len(cur_ids)):
            depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]




        depth_energy = depth_factor*np.sqrt(np.sum((depths - depths_curpos)**2))

        return depth_energy

    def get_V_matrix(self, mds_dist_mat,W):
        """ Get V matrix considering weights. This is for
        reducing the impact of distant forces.

        Args:
            mds_dist_mat:

        Returns:

        """
        m,N = np.shape(mds_dist_mat)

        V = -copy.deepcopy(W)
        for i in range(N):
            V[i,i] = -np.sum(V[i,:])

        # V = np.ones((N,N))
        # V *= -1
        # np.fill_diagonal(V,N-1)

        return V

    def get_B_matrix(self,N, mds_dist_mat, cur_dist_mat,W):
        """Construct B matrix from http://tosca.cs.technion.ac.il/book/handouts/Stanford09_mds.pdf

        Args:
            mds_dist_mat: Ideal distance matrix.
            cur_dist_mat: Current distance matrix.

        Returns:

        """

        B = np.zeros((N,N))

        for i in range(N):
            for j in range(N):
                if i != j and cur_dist_mat[i,j]>np.finfo(float).eps:
                    B[i,j] = -W[i,j]*mds_dist_mat[i,j]/cur_dist_mat[i,j]
                    # B[i,j] = -mds_dist_mat[i,j]/cur_dist_mat[i,j]
                elif i!=j:
                    B[i,j] = 0

        for i in range(N):
            B[i,i] = -(np.sum(B[i,:]) - B[i,i])


        return B

    def center_about_median(self, cur_pos, cur_median_ind):
        """

        Args:
            cur_pos:

        Returns:

        """

        median_pos = copy.deepcopy(cur_pos[cur_median_ind])

        for i in range(len(cur_pos)):
            cur_pos[i] -= median_pos

        return cur_pos

    def compute_depth_spline(self,cur_pos, depths, cur_median_ind):
        """Compute the interpolation spline after computing the depths for the 2D space.

        Returns:

        """
        # # compute depths based on cur_pos
        # analyser = hsa.analyzer(members=cur_pos, grid_members=None, kernel=None) # gridpoints2d is vestigial here
        # depths_curpos = analyser.get_depths_extern()
        # depths_curpos = depths_curpos - np.float(np.min(depths_curpos))
        # depths_curpos = depths_curpos/np.float(np.max(depths_curpos))

        # G = linear_kernel(cur_pos)
        #
        # analyser_spa = spa.SpatialDepth(G)
        # depths_curpos = analyser_spa.get_depths_from_gram()
        # dprint(np.amax(depths_curpos),np.amin(depths_curpos))
        # depths_curpos = self.depths_scaler(depths_curpos, orig_depths_scale, orig_depths_min) # scaling the range of depth_curpos to match range of depth
        # depths_curpos = depths


        # depths_curpos = self.depths_scaler(depths_curpos,1,0)
        #
        # sorted_inds = sorted(range(depths_curpos.shape[0]), key=lambda k: depths_curpos[k])
        # median_pos = copy.deepcopy(cur_pos[sorted_inds[-1]])
        median_pos = copy.deepcopy(cur_pos[cur_median_ind])


        max_dist_from_median = 0
        for i in range(len(cur_pos)):
            cur_pos[i] -= median_pos
            cur_dist = np.linalg.norm(cur_pos[i])

            if cur_dist > max_dist_from_median:
                max_dist_from_median = cur_dist

        # compute the tps spline
        xs = [pt[0] for pt in cur_pos]
        ys = [pt[1] for pt in cur_pos]


        tps = Tps(xs, ys, depths)
        # tps = Rbf(xs,ys,depths,function='linear')

        # tps = Tps(xs, ys, depths_curpos)
        # polarize
        phis = np.linspace(-np.pi, np.pi, self.res)
        rhos = np.linspace(0, max_dist_from_median*1.1, self.res)
        # rhos = np.linspace(0, 1 , self.res)

        PI, RI = np.meshgrid(phis, rhos)
        ZI_pol = misc.get_depths_for_polar_grid(PI,RI,tps)

        # monotonize polarized spline
        monofit = monocol.MonotonicityColwise(RI, ZI_pol)
        ZI_pol_mono = monofit(bw=self.bw)
        # ZI_pol_mono = ZI_pol

        # depolarize spline (conversion to cartesian coordinates)
        PI_flat = PI.flatten()
        RI_flat = RI.flatten()
        X_flat = np.empty_like(PI_flat)
        Y_flat = np.empty_like(PI_flat)
        X_flat = X_flat.reshape((len(X_flat),1))
        Y_flat = Y_flat.reshape((len(Y_flat),1))
        for i in range(len(PI_flat)):
            xc,yc = misc.pol2cart(RI_flat[i], PI_flat[i])
            X_flat[i] = xc
            Y_flat[i] = yc

        xmax_car = np.amax(X_flat)
        ymax_car = np.amax(Y_flat)
        xmin_car = np.amin(X_flat)
        ymin_car = np.amin(Y_flat)
        xs2 = np.linspace(xmin_car,xmax_car,self.res)
        ys2 = np.linspace(ymin_car,ymax_car,self.res)
        XI2,YI2 = np.meshgrid(xs2,ys2)



        ZI_pol_mono_flat = ZI_pol_mono.flatten()
        ZI_pol_mono_flat = ZI_pol_mono_flat.reshape(np.shape(X_flat))

        ZI_car_mono = griddata(np.hstack((X_flat,Y_flat)), ZI_pol_mono_flat, (XI2, YI2), method='cubic')
        ZI_car_mono = ZI_car_mono.reshape((self.res,self.res))
        ZIcm = np.nan_to_num(ZI_car_mono) # just making the name shorter for convinience.


        # exit(0)

        return cur_pos,XI2,YI2,ZIcm

    def get_dist_forces(self,mds_dist_mat, cur_dist_mat, cur_pos, k=1):
        """Computes the force vectors due to the stresses.

        Args:
            mds_dist_mat (2d np array): Ideal distance matrix.
            cur_dist_mat (2d np array): Current distance matrix.
            cur_pos_2d (list of 1d np arrays): current positions to generate the stress caused force direction.
            k(int): spring constant.

        Returns:
            dist_forces(list): A list of vectors denoting forces due to the stress of deviation from ideal distances.

        """


        N = len(cur_pos)

        W = np.reciprocal(mds_dist_mat)
        W = W*W
        np.fill_diagonal(W, 0)

        Z = np.array(cur_pos)


        B = self.get_B_matrix(N,mds_dist_mat, cur_dist_mat,W)

        V = self.get_V_matrix(mds_dist_mat, W)

        grad_sig = 2*(np.dot(V,Z)-np.dot(B,Z))


        SCAMOF_FAC = float(1)/(2*len(cur_pos)) # denominator for scamof gradient descent
        dfs = SCAMOF_FAC*grad_sig

        return dfs

    def get_total_energy(self, ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb, cur_pos, depth_factor):
        """Computes and returns the total energy.

        Args:
            ZIcm:
            XI2:
            YI2:
            cur_pos:

        Returns:

        """

        # cur_dist_mat = self.list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
        # dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
        #
        # cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res, XI2, YI2)
        # depths_curpos = np.zeros(len(cur_ids))
        #
        # for i in range(len(cur_ids)):
        #     depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]
        #
        # depth_energy = np.sum((self.depths - depths_curpos)**2)
        #

        N = len(cur_pos)

        # depth energy 1
        cur_pos1 = copy.deepcopy(cur_pos[:self.class_boun])
        cur_pos1_centered = self.center_about_median(cur_pos1, self.ori_median_ind1)
        depth_energy1 = self.get_depth_energy(self.depths1, ZIcm_a, XIa, YIa, cur_pos1_centered, depth_factor)
        depth_energy1 /= self.class_boun

        # depth energy 2
        cur_pos2 = copy.deepcopy(cur_pos[self.class_boun:])
        cur_pos2_centered = self.center_about_median(cur_pos2, self.ori_median_ind2)
        depth_energy2 = self.get_depth_energy(self.depths2, ZIcm_b, XIb, YIb, cur_pos2_centered, depth_factor)
        depth_energy2 /= N - self.class_boun

        # self.original_dist_mat = self.list_to_dist_mat(original_points)
        cur_dist_mat = self.list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
        dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
        dis_energy /= N**2

        return dis_energy+depth_energy1+depth_energy2

    def total_linesearch(self, ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb, gd, cur_pos, rho, c):
        """

        Args:
           ZIcm:
           XI2:
           YI2:
           gd: gradient of the total force
           cur_pos: list of 2d np array
           rho:
           c:

        Returns:

        """
        alphak=1
        # compute fk and gk
        gk = np.array(gd)


        # store orig x
        xx = np.array(cur_pos)
        fk = self.get_total_energy(ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb,xx)
        x = copy.deepcopy(xx)

        # while np.linalg.norm(alphak*gk) > self.max_step:
        #     alphak = alphak*rho
        x = x - alphak*gk

        # compute updated x with original alpha
        fk1 = self.get_total_energy(ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb, x)
        # dprint(fk1, fk - c*alphak*np.sum(np.multiply(gk,gk)), gk)
        # while undesirable condition exists
        while fk1 > fk - c*alphak*np.sum(np.multiply(gk,gk)):
            # compute the newer alpha
            alphak = alphak*rho
            # compute the newer x using the new alpha
            x = xx - alphak*gk
            # compute the new objective function value at x
            fk1 = self.get_total_energy(ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb, x)
            # dprint(fk1, fk, c*alphak*np.sum(np.multiply(gk,gk)), alphak)
        # dprint(alphak)
        return alphak

    def get_total_forces(self, mds_dist_mat, cur_dist_mat, cur_pos, XIa, YIa, ZIcm_a, XIb, YIb, ZIcm_b, cur_pos1, cur_pos2,fixed_stepsize, depth_factor, iter):
        """Computes the total force for the next iteration based on the total energy and
        adaptive stepsize.

        Returns:

        """



        # get dist forces gradient
        N = len(cur_pos)
        W = np.reciprocal(mds_dist_mat)
        W = W*W
        np.fill_diagonal(W, 0)
        # np.set_printoptions(threshold=np.nan)

        # W[W==np.Inf] = 10e15 # # dealing with overlapping points

        Z = np.array(cur_pos)
        # V = -1*np.ones((N,N))
        # np.fill_diagonal(V, N-1)
        B = self.get_B_matrix(N,mds_dist_mat, cur_dist_mat,W)
        V = self.get_V_matrix(mds_dist_mat,W)

        grad_sig = 2*(np.dot(V,Z)-np.dot(B,Z))
        grad_sig /= N**2


        # get depth forces gradient

        # cur_pos1 = cur_pos[:self.class_boun]
        # cur_pos2 = cur_pos[self.class_boun:]



        # compute gradients at the point positions in the cur_ids using central differences
        gd = []
        diffs = np.zeros(N)

        # first part
        cur_ids = ipos.get_pos_on_grid2(cur_pos1, self.res, XIa, YIa)
        for j in range(len(cur_ids)):
            ix = cur_ids[j][1];iy = cur_ids[j][0] # swapped ix and iy to make consistent with np indexing/fetching
            gy = (ZIcm_a[ix+1,iy] - ZIcm_a[ix-1,iy])/self.twice_grid_side
            gx = (ZIcm_a[ix,iy+1] - ZIcm_a[ix,iy-1])/self.twice_grid_side
            gd.append(np.array([gx,gy]))
            diffs[j] = ZIcm_a[ix,iy]-self.depths1[j]



        # second part
        cur_ids = ipos.get_pos_on_grid2(cur_pos2,self.res,XIb,YIb)
        js = range(N-self.class_boun)
        for j in js:
            ix = cur_ids[j][1];iy = cur_ids[j][0] # swapped ix and iy to make consistent with np indexing/fetching
            gy = (ZIcm_b[ix+1,iy] - ZIcm_b[ix-1,iy])/self.twice_grid_side
            gx = (ZIcm_b[ix,iy+1] - ZIcm_b[ix,iy-1])/self.twice_grid_side
            gd.append(np.array([gx,gy]))
            diffs[j+self.class_boun] = ZIcm_b[ix,iy]-self.depths2[j]
            # if j==42:
                # dprint(gx,gy)
                # dprint(ZIcm_b[ix,iy], self.depths2[j], ZIcm_b[ix,iy]-self.depths2[j])
                # diffs[j+self.class_boun]=0
            #
            #     exit(0)
        diffs[self.ori_median_ind1] = 0
        diffs[self.class_boun+self.ori_median_ind2] = 0

        gd = np.array(gd)
        grad_dep = 2*np.multiply(gd, diffs[:,np.newaxis])
        grad_dep /= N
        grad_dep *= depth_factor

        gd_total = grad_sig + grad_dep

        # get the alpha


        # compute the step using alpha and the total gradient
        # alpha = 0.01
        # alpha = 1
        alpha = fixed_stepsize
        # if fixed_stepsize is not None:
        #     alpha = fixed_stepsize
        # else:
        #     # alpha = self.total_linesearch(ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb,gd_total, cur_pos, rho=0.5, c=0.0001)
        #     if self.adaptive_enabled:
        #         alpha = self.total_linesearch(ZIcm_a, XIa, YIa, ZIcm_b, XIb, YIb,gd_total, cur_pos,rho=0.5, c=0.0001)
        #         # alpha = fixed_stepsize
        #     else:
        #         alpha = 1
        #     # dprint('alpha', alpha)

        # return the step
        tfs = alpha*gd_total



        return tfs, grad_dep, grad_sig

    def get_mds_dc(self, original_points, cur_pos, show_fig, spline_lag,adap_starts_after, disable_fixed_step_after,
                   depth_factor, orig_dist_mat=None, fixed_stepsize=None, alpha=None):
        """

        Args:
            original_points:
            cur_pos:
            show_fig:
            spline_lag:

        Returns:

        """
        N=len(cur_pos)

        energies = [[],[],[],[]]

        cur_pos1 = copy.deepcopy(cur_pos[:self.class_boun])
        cur_pos2 = copy.deepcopy(cur_pos[self.class_boun:])

        analyser = hsa.analyzer(members=cur_pos1, grid_members=None, kernel=None) # gridpoints2d is vestigial here
        depths_curpos1 = analyser.get_depths_extern()
        orig_one_inds1 = np.where(self.depths1==1)
        hs_for_ones1 = depths_curpos1[orig_one_inds1]
        self.ori_median_ind1 = orig_one_inds1[0][np.argmax(hs_for_ones1)]

        analyser = hsa.analyzer(members=cur_pos2, grid_members=None, kernel=None) # gridpoints2d is vestigial here
        depths_curpos2 = analyser.get_depths_extern()
        orig_one_inds2 = np.where(self.depths2==1)
        hs_for_ones2 = depths_curpos2[orig_one_inds2]
        self.ori_median_ind2 = orig_one_inds2[0][np.argmax(hs_for_ones2)]

        median, band50_a, band100_a, outliers, cat_list_pre, band_prob_bounds1 = get_median_and_bands(self.depths1, alpha=alpha)
        median, band50_b, band100_b, outliers, cat_list_pre, band_prob_bounds2 = get_median_and_bands(self.depths2, alpha=alpha)

        dprint(band_prob_bounds1)
        dprint(band_prob_bounds2)

        N1 = len(cur_pos1)
        N2 = len(cur_pos2)


        # get the energies at the begining.
        cur_pos1_centered,XIa,YIa,ZIcm_a = self.compute_depth_spline(cur_pos1, self.depths1, self.ori_median_ind1)
        self.old_ZIcm_a = ZIcm_a


        depth_energy1 = self.get_depth_energy(self.depths1, ZIcm_a, XIa, YIa, cur_pos1_centered, depth_factor)
        depth_energy1 /= N

        cur_pos2_centered,XIb,YIb,ZIcm_b = self.compute_depth_spline(cur_pos2, self.depths2, self.ori_median_ind2)
        self.old_ZIcm_b = ZIcm_b
        depth_energy2 = self.get_depth_energy(self.depths2, ZIcm_b, XIb, YIb, cur_pos2_centered, depth_factor)
        depth_energy2 /= N



        energies[0].append(depth_energy1)
        energies[1].append(depth_energy2)

        # self.original_dist_mat = self.list_to_dist_mat(original_points)

        if original_points is not None:
            self.original_dist_mat = self.list_to_dist_mat(original_points)
        else:
            self.original_dist_mat = orig_dist_mat


        cur_dist_mat = self.list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
        dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
        dis_energy/=N**2

        energies[2].append(dis_energy)

        energies[3].append(depth_energy1+depth_energy2+dis_energy)

        iter_at_last_spline_compute = 0




        data_dict = {}
        data_dict['orig_depths1'] = list(self.depths1)
        data_dict['max_orig_depth1'] = np.amax(self.depths1)
        data_dict['min_orig_depth1'] = np.amin(self.depths1)
        data_dict['orig_depths2'] = list(self.depths2)
        data_dict['max_orig_depth2'] = np.amax(self.depths2)
        data_dict['min_orig_depth2'] = np.amin(self.depths2)
        data_dict['spline_lag'] = spline_lag
        data_dict['N_i'] = self.N_i;
        data_dict['median1_ind'] = self.ori_median_ind1
        data_dict['median2_ind'] = self.class_boun+self.ori_median_ind2
        data_dict['orig_depths'] = list(self.depths1)+list(self.depths2)
        data_dict['class_boun'] = self.class_boun
        data_dict['ori_median1_ind'] = self.ori_median_ind1
        data_dict['ori_median2_ind'] = self.ori_median_ind2
        dprint(self.ori_median_ind2)

        data_dict['band50_a'] = list(band50_a)
        data_dict['band100_a'] = list(band100_a)
        data_dict['band50_b'] = [x+self.class_boun for x in list(band50_b)]
        data_dict['band100_b'] = [x+self.class_boun for x in list(band100_b)]

        with open("output_tsvs/orig_depths.json", 'w') as outfile:
            json.dump(data_dict, outfile, sort_keys=True, indent=4)


        for iter in range(self.N_i):

            # performing the unconstrained step.
            cur_dist_mat = self.list_to_dist_mat(cur_pos)
            dis_forces = self.get_dist_forces(mds_dist_mat=self.original_dist_mat, cur_dist_mat=cur_dist_mat, cur_pos=cur_pos, k=1)
            # dfs = np.array(dis_forces)

            # dprint(cur_pos)

            if iter%spline_lag==0:

                cur_pos1 = copy.deepcopy(cur_pos[:self.class_boun])
                cur_pos1_centered,XIa,YIa,ZIcm_a = self.compute_depth_spline(cur_pos1, self.depths1, self.ori_median_ind1)
                ZIcm_a = (ZIcm_a + self.old_ZIcm_a)/2
                self.old_ZIcm_a = ZIcm_a

                cur_pos2 = copy.deepcopy(cur_pos[self.class_boun:])

                dprint(self.ori_median_ind2)
                cur_pos2_centered,XIb,YIb,ZIcm_b = self.compute_depth_spline(cur_pos2, self.depths2, self.ori_median_ind2)
                ZIcm_b = (ZIcm_b + self.old_ZIcm_b)/2
                self.old_ZIcm_b = ZIcm_b


                ZIcm_a_max = np.amax(ZIcm_a)
                ZIcm_a_min = np.amin(ZIcm_a)
                ZIcm_b_max = np.amax(ZIcm_b)
                ZIcm_b_min = np.amin(ZIcm_b)


                ZIcm_a_linear = np.ndarray.flatten(np.flipud(ZIcm_a))
                ZIcm_b_linear = np.ndarray.flatten(np.flipud(ZIcm_b))
                data_dict = {}

                data_dict["field_a"] = np.around(ZIcm_a_linear, decimals=3).tolist()
                data_dict["field_b"] = np.around(ZIcm_b_linear, decimals=3).tolist()
                data_dict['maxx_a'] = round(XIa[-1,-1],3)
                data_dict['minx_a'] = round(XIa[0,0],3)
                data_dict['maxy_a'] = round(YIa[-1,-1],3)
                data_dict['miny_a'] = round(YIa[0,0],3)

                data_dict['maxx_b'] = round(XIb[-1,-1],3)
                data_dict['minx_b'] = round(XIb[0,0],3)
                data_dict['maxy_b'] = round(YIb[-1,-1],3)
                data_dict['miny_b'] = round(YIb[0,0],3)


                jsonname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.json'
                with open(jsonname, 'w') as outfile:
                    json.dump(data_dict, outfile, sort_keys=True, indent=4)

                # cur_pos_centered = cur_pos1_centered+cur_pos2_centered

                if iter<disable_fixed_step_after:
                    self.adaptive_enabled = False

                iter_at_last_spline_compute = iter


                 # splines images
                fig = plt.figure(frameon=False)
                fig.set_size_inches(w=4,h=4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.axis('equal')
                # plt.pcolor(XI2, YI2, ZIcm, cmap=cm.YlGnBu, vmin=0, vmax=1)

                red_map = cm.YlGnBu
                red_map.set_under('k', alpha=0)

                ZIcm_a_masked = np.ma.masked_array(ZIcm_a, ZIcm_a<0.00001)
                plt.pcolor(XIa, YIa, ZIcm_a, cmap=red_map, vmin=0.00001, vmax=1)
                # levels = np.arange(0.000001, 1, 0.1)
                # CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                # CS = plt.contour(XIa, YIa, ZIcm_a,alpha=1.0, linewidths=0.1, levels=levels, colors='k',
                #                  antialiased=True)
                # plt.clabel(CS, inline=1, fontsize=8)
                plt.axis('off')

                imname ='output_tsvs/spline_only_a_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                fig.savefig(imname, dpi=100)
                im = Image.open(imname)
                im.putalpha(192)
                im.save(imname)
                plt.close()

                fig = plt.figure(frameon=False)
                fig.set_size_inches(w=4,h=4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.axis('equal')
                # plt.pcolor(XI2, YI2, ZIcm, cmap=cm.YlGnBu, vmin=0, vmax=1)
                blue_map = cm.YlGnBu
                blue_map.set_under('k', alpha=0)

                ZIcm_b_masked = np.ma.masked_array(ZIcm_b, ZIcm_b<0.00001)

                plt.pcolor(XIb, YIb, ZIcm_b, cmap=blue_map, vmin=0.00001, vmax=1)
                # CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                # CS = plt.contour(XIb, YIb, ZIcm_b,alpha=1.0, linewidths=0.1, levels=levels, colors='k',
                #                  antialiased=True)
                # plt.clabel(CS, inline=1, fontsize=8)
                plt.axis('off')

                imname ='output_tsvs/spline_only_b_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                fig.savefig(imname, dpi=100)
                im = Image.open(imname)
                im.putalpha(192)
                im.save(imname)
                plt.close()



            else:
                # recompute the cur_pos1_centered and cur_pos2_centered but not spline
                cur_pos1 = copy.deepcopy(cur_pos[:self.class_boun])
                cur_pos1_centered = self.center_about_median(cur_pos1, self.ori_median_ind1)

                cur_pos2 = copy.deepcopy(cur_pos[self.class_boun:])
                cur_pos2_centered = self.center_about_median(cur_pos2, self.ori_median_ind2)



            if iter - iter_at_last_spline_compute == adap_starts_after:
                self.adaptive_enabled=True

            tfs,dfgd,dgd = self.get_total_forces(self.original_dist_mat, cur_dist_mat,cur_pos,XIa,YIa, ZIcm_a,XIb,YIb, ZIcm_b, cur_pos1_centered, cur_pos2_centered,fixed_stepsize, depth_factor, iter)


            #vis
            xs = [pos[0] for j,pos in enumerate(cur_pos)] # after centering the cur_pos about median
            ys = [pos[1] for j,pos in enumerate(cur_pos)]

            arrow_scale = 1

            plt.figure()

            F = plt.gcf()
            DefaultSize = F.get_size_inches()
            F.set_size_inches((DefaultSize[0]*3, DefaultSize[1]*2)) # http://scipy-cookbook.readthedocs.io/items/Matplotlib_AdjustingImageSize.html

            plt.suptitle('Iteration: '+str(iter))


            xs_c1 = [pos[0] for j,pos in enumerate(cur_pos1)]
            ys_c1 = [pos[1] for j,pos in enumerate(cur_pos1)]



            # plt.subplot(2,4,1)
            # plt.figure()
            # plt.axis('equal')
            # plt.title('MonoTPS C1')
            # plt.pcolor(XIa, YIa, ZIcm_a, cmap=cm.autumn_r, vmin=ZIcm_a_min, vmax=ZIcm_a_max)
            # plt.scatter(xs_c1, ys_c1, 30, self.depths1, cmap=cm.autumn_r, vmin=np.amin(ZIcm_a), vmax=np.amax(ZIcm_a))
            # plt.colorbar()
            # plt.savefig('output_tsvs/spline_'+format(iter, '04')+'.png')
            #
            # xs_c2 = [pos[0] for j,pos in enumerate(cur_pos2)]
            # ys_c2 = [pos[1] for j,pos in enumerate(cur_pos2)]
            #
            # plt.subplot(2,4,2)
            # plt.figure()
            # plt.axis('equal')
            # plt.title('MonoTPS C2')
            # plt.pcolor(XIb, YIb, ZIcm_b, cmap=cm.winter_r, vmin=ZIcm_b_min, vmax=ZIcm_b_max)
            # plt.scatter(xs_c2, ys_c2, 30, self.depths2, cmap=cm.winter_r, vmin=np.amin(ZIcm_b), vmax=np.amax(ZIcm_b))
            # plt.colorbar()
            # plt.savefig('output_tsvs/spline2_'+format(iter, '04')+'.png')
            #
            # a=plt.subplot(2,4,3)
            # # plt.axis('equal')
            #
            # a.set_ylim([-0.7,0.7])
            # a.set_xlim([-0.7,0.7])
            # plt.title('Forces: dis(red), dep(green)')
            # # plt.scatter(xs, ys, 30, self.depths, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
            # # plt.colorbar()
            # ax = plt.gca()
            # ax.quiver(xs, ys,  -dfgd[:,0], -dfgd[:,1], angles='xy',scale=arrow_scale,color='g')
            # ax.quiver(xs, ys,  -dgd[:,0], -dgd[:,1], angles='xy',scale=arrow_scale,color='r')
            #
            # for i2 in range(len(xs)):
            #     ax.annotate(str(i2), (xs[i2],ys[i2]))

            # if iter%spline_lag==0:



                # jsonname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.json'
                # with open(jsonname, 'w') as outfile:
                #     json.dump(data_dict, outfile, sort_keys=True, indent=4)

                # # Now drawing spline only
                #
                # fig = plt.figure(frameon=False)
                # fig.set_size_inches(w=4,h=4)
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # plt.axis('equal')
                # # plt.pcolor(XIa, YIa, ZIcm_a, cmap=cm.autumn_r, vmin=ZIcm_a_min, vmax=ZIcm_a_max)
                # levels = np.arange(0, 1, 0.1)
                # # CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                # CS = plt.contour(XIa, YIa, ZIcm_a,alpha=1.0, linewidths=0.1, levels=levels, colors='k',
                #                  antialiased=True)
                # # plt.clabel(CS, inline=1, fontsize=8)
                # plt.axis('off')
                #
                # imname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                # fig.savefig(imname, dpi=100)
                # im = Image.open(imname)
                # im.putalpha(255)
                # im.save(imname)
                # plt.close()
                #
                # fig = plt.figure(frameon=False)
                # fig.set_size_inches(w=4,h=4)
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # plt.axis('equal')
                # # plt.pcolor(XIb, YIb, ZIcm_b, cmap=cm.winter_r, vmin=ZIcm_a_min, vmax=ZIcm_a_max)
                # levels = np.arange(0, 1, 0.1)
                # # CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                # CS = plt.contour(XIb, YIb, ZIcm_b,alpha=1.0, linewidths=0.1, levels=levels, colors='k',
                #                  antialiased=True)
                # # plt.clabel(CS, inline=1, fontsize=8)
                # plt.axis('off')
                #
                # imname ='output_tsvs/spline2_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                # fig.savefig(imname, dpi=100)
                # im = Image.open(imname)
                # im.putalpha(128)
                # im.save(imname)
                # plt.close()



            norms = np.linalg.norm(tfs, axis=1)
            pos = np.where(norms>self.max_step)
            tfs[pos] = (tfs[pos]/norms[pos,None])*self.max_step


            for j in range(len(cur_pos)):
                cur_pos[j] = cur_pos[j] - tfs[j]



            cur_dist_mat = self.list_to_dist_mat(cur_pos)
            dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
            dis_energy/=N**2
            energies[2].append(dis_energy)

            # computing depth energies
            cur_pos1 = cur_pos[:self.class_boun]
            cur_pos2 = cur_pos[self.class_boun:]
            # get the energies at the begining.
            # cur_pos1,XIa,YIa,ZIcm_a_del, depths_curpos = self.compute_depth_spline(cur_pos1, self.depths1_range, self.depths1_min)
            # self.old_ZIcm_a = ZIcm_a
            depth_energy1 = self.get_depth_energy(self.depths1, ZIcm_a, XIa, YIa, cur_pos1_centered, depth_factor)
            depth_energy1 /= N

            # cur_pos2,XIb,YIb,ZIcm_b_del, depths_curpos = self.compute_depth_spline(cur_pos2, self.depths2_range, self.depths2_min)
            # self.old_ZIcm_b = ZIcm_b
            depth_energy2 = self.get_depth_energy(self.depths2, ZIcm_b, XIb, YIb, cur_pos2_centered, depth_factor)
            depth_energy2 /= N

            energies[0].append(depth_energy1)
            energies[1].append(depth_energy2)
            energies[3].append(depth_energy1+depth_energy2+dis_energy)

            # forces[0].append(depth_energy1+depth_energy2)

            dprint(iter)


            xs_new = [pos[0] for j,pos in enumerate(cur_pos)]
            ys_new = [pos[1] for j,pos in enumerate(cur_pos)]

            a=plt.subplot(2,4,4)
            a.set_ylim([-0.7,0.7])
            a.set_xlim([-0.7,0.7])
            # plt.axis('equal')
            plt.title('New positions')
            plt.scatter(xs_new[:self.class_boun], ys_new[:self.class_boun], 30, self.depths1, cmap=cm.autumn_r, vmin=np.amin(ZIcm_a), vmax=np.amax(ZIcm_a))
            plt.colorbar()
            plt.scatter(xs_new[self.class_boun:], ys_new[self.class_boun:], 30, self.depths2, cmap=cm.winter_r, vmin=np.amin(ZIcm_b), vmax=np.amax(ZIcm_b))
            plt.colorbar()
            # plt.scatter(xs_new, ys_new, 30)
            ax = plt.gca()

            for i2 in range(len(xs)):
                ax.annotate(str(i2), (xs_new[i2],ys_new[i2]))


            plt.subplot(2,4,5)
            # plt.axis('equal')
            plt.figure()
            plt.title('Depth energy 1')
            # plt.plot(range(iter+2),forces[2],'black',label="total")
            # plt.plot(range(1+(iter+1)*(1+self.M_i)),forces[0],'g--',label="depth")
            plt.plot(range(2+(iter)),energies[0],'g--')
            # plt.plot(range(1+(iter)*(self.M_i+1)+iter_M+2),scaled_f1s,'r--',label="dist")
            plt.legend(loc=2)
            ax = plt.gca()
            maxe0 = max(energies[0])
            ax.set_ylim([-maxe0*0.1,maxe0*1.1])
            plt.savefig('output_tsvs/dep1_'+format(iter, '04')+'.png')

            plt.subplot(2,4,6)
            # plt.axis('equal')
            plt.figure()
            plt.title('Depth energy 2')
            # plt.plot(range(iter+2),forces[2],'black',label="total")
            # plt.plot(range(1+(iter+1)*(1+self.M_i)),forces[0],'g--',label="depth")
            plt.plot(range(2+(iter)),energies[1],'g--')
            # plt.plot(range(1+(iter)*(self.M_i+1)+iter_M+2),scaled_f1s,'r--',label="dist")
            plt.legend(loc=2)
            ax = plt.gca()
            maxe1 = max(energies[1])
            ax.set_ylim([-maxe1*0.1,maxe1*1.1])
            plt.savefig('output_tsvs/dep2_'+format(iter, '04')+'.png')

            # total_energy = [x + y for x, y in zip(forces[0], forces[1])]

            plt.subplot(2,4,7)
            # plt.axis('equal')
            plt.figure()
            plt.title('Stress (dist) energy')
            # plt.plot(range(iter+2),forces[2],'black',label="total")
            # plt.plot(range(1+(iter+1)*(1+self.M_i)),forces[0],'g--',label="depth")
            # plt.plot(range(1+(iter)*(self.M_i+1)+iter_M+2),scaled_f0s,'g--',label="depth")
            plt.plot(range(2+(iter)),energies[2],'r--')
            plt.legend(loc=2)
            ax = plt.gca()
            maxe2 = max(energies[2])
            ax.set_ylim([-maxe2*0.1,maxe2*1.1])
            plt.savefig('output_tsvs/dis_'+format(iter, '04')+'.png')

            plt.subplot(2,4,8)
            # plt.axis('equal')
            plt.figure()
            plt.title('Total energy')
            # plt.plot(range(iter+2),forces[2],'black',label="total")
            # plt.plot(range(1+(iter+1)*(1+self.M_i)),forces[0],'g--',label="depth")
            plt.plot(range(2+(iter)),energies[3],'b--')
            # plt.plot(range(1+(iter)*(self.M_i+1)+iter_M+2),scaled_f1s,'r--',label="dist")
            plt.legend(loc=2)
            ax = plt.gca()
            maxe3 = max(energies[3])
            ax.set_ylim([-maxe3*0.1,maxe3*1.1])
            plt.savefig('output_tsvs/tot_'+format(iter, '04')+'.png')

            # np.savez('output_jsons/'+format(iter, '04')+'.npz',ZIcm_a,ZIcm_b)

            # showFig = False
            if show_fig:
                plt.show(block=False)
            else:
                # plt.savefig('output/'+format(iter, '04')+'.png', bbox_inches='tight')
                # plt.savefig('output/'+format(iter, '04')+'.png')

                with open('output_tsvs/'+format(iter, '04')+'.tsv', 'wb') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    header = ['cpx', 'cpy', 'npx', 'npy', 'dpfx', 'dpfy', 'dfx', 'dfy','oridep']
                    writer.writerow(header)
                    for i in range(N1):
                        row = [round(xs[i],4), round(ys[i],4), round(xs_new[i],4), round(ys_new[i],4),
                               round(dfgd[i,0],4), round(dfgd[i,1],4), round(dgd[i,0],4), round(dgd[i,1],4),
                               round(self.depths1[i],4)]
                        writer.writerow(row)

                    for i in range(N2):
                        j = i+N1
                        row = [round(xs[j],4), round(ys[j],4), round(xs_new[j],4), round(ys_new[j],4),
                               round(dfgd[j,0],4), round(dfgd[j,1],4), round(dgd[j,0],4), round(dgd[j,1],4),
                               round(self.depths1[i],4)]
                        writer.writerow(row)

                plt.close()




            # if iter==1:
            #     exit(0)
        if show_fig:
            plt.show()

        return cur_pos


# todo -
# - write figure backend and front end, write json for each vis for later exploration, http://mpld3.github.io/ # http://zynga.github.io/scroller/
# - adaptive energy for unimodal and multimodal
# - get bimodal working**
# - 2d html viewer for all the vis with tooltip interaction**

# Change MDS to regular.
