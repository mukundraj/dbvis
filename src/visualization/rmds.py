'''
Created on Mar 31, 2017

Radial constrained mds for graph layout.

diffs from dc_mds:
- no more depth matching, only spline interpolation and monotonization

todo:
- coordinate descent
-

@author: mukundraj
'''
from produtils import dprint
import math
import matplotlib.pyplot as plt
import numpy as np
from produtils import dprint
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
from datarelated.processing.depths import get_median_and_bands
from sklearn.metrics.pairwise import linear_kernel
import src.analysis.spatial_analyzer as spa
import matplotlib
from PIL import Image
import src.analysis.mahalanobis_analyzer as msa
from sklearn.neighbors import KernelDensity
from scipy import interpolate
from scipy import optimize


class rmds(object):
    """

    """

    def __init__(self, depths, N_i, M_i,res,bw, max_step, num_old, num_old_spline, depth_type, alpha=None, adj_mat=None):
        """

        Returns:

        """
        self.depth_type = depth_type
        self.alpha = alpha

        self.depths = depths
        self.depths_max = np.amax(self.depths)
        self.depths_min = np.amin(self.depths)
        self.depths_range = self.depths_max - self.depths_min
        self.N_i = N_i
        self.M_i = M_i
        self.res = res
        self.bw = bw
        self.max_step = max_step
        self.twice_grid_side = float(2)/res
        # self.gd = None # container to store the gradient to avoid recomputation of gradient.

        self.old_ZIcm = None

        self.old_ZIcm_s = np.zeros((res,res,num_old_spline))

        self.num_old = num_old
        self.num_old_spline = num_old_spline

        self.old_poses = np.zeros((len(depths),2*num_old))


        self.adaptive_enabled = False
        self.running_avg_enabled = False

        self.A = adj_mat

    def depths_scaler(self, c_depths):
        """Scales the current depths to have the same range as the input depths.

        Args:
            c_depths: Current depths.

        Returns:

        """
        c_depths_min = np.amin(c_depths)
        c_depths_max = np.amax(c_depths)
        # dprint('maxmin', c_depths_max,c_depths_min)
        ratio = self.depths_range/(c_depths_max - c_depths_min)

        c_depths = c_depths - c_depths_min
        c_depths = c_depths*ratio + self.depths_min
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
                dist_mat[i,j] = distance.euclidean(pos[i],pos[j])
                dist_mat[j,i] = dist_mat[i,j]

        return dist_mat



    def get_depth_engergy(self,  ZIcm, XI2, YI2, cur_pos, depth_factor):
        """Get the curre

        Args:
            ZIcm (2d array):
            cur_ids (list):

        Returns:
            depth_energy (float):

        """

        # dprint(distance.euclidean(cur_pos[1],cur_pos[2]))
        cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res, XI2, YI2)
        depths_curpos = np.zeros(len(cur_ids))

        for i in range(len(cur_ids)):
            depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]

        depth_energy = depth_factor*np.sum((self.depths - depths_curpos)**2)

        return depth_energy


    def get_V_matrix(self, mds_dist_mat, W):
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

    def get_B_matrix(self,N, mds_dist_mat, cur_dist_mat, W):
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






    def compute_depth_spline(self,cur_pos, iter=None):
        """Compute the interpolation spline after computing the depths for the 2D space.

        Returns:

        """
        # compute depths based on cur_pos

        if self.depth_type==0: # half space
            analyser = hsa.analyzer(members=cur_pos, grid_members=None, kernel=None) # gridpoints2d is vestigial here
            depths_curpos = analyser.get_depths_extern()

        elif self.depth_type==1:
            G = linear_kernel(cur_pos)
            analyser_spa = spa.SpatialDepth(G)
            depths_curpos = analyser_spa.get_depths_from_gram()
        elif self.depth_type==2:
            analyser = msa.MahaDepth(members=cur_pos)
            depths_curpos = analyser.get_depths()


        depths_curpos = self.depths_scaler(depths_curpos) # scaling the range of depth_curpos to match range of depth
        # self.depths = self.histmatch(self.depths, depths_curpos, bw=0.02,showplot=True)

        # sorted_inds = sorted(range(depths_curpos.shape[0]), key=lambda k: depths_curpos[k])

        sorted_inds = sorted(range(self.depths.shape[0]), key=lambda k: self.depths[k])

        self.ori_median_ind = sorted_inds[-1]
        median_pos = copy.deepcopy(cur_pos[sorted_inds[-1]])



        max_dist_from_median = 0
        for i in range(len(cur_pos)):
            cur_pos[i] -= median_pos
            cur_dist = np.linalg.norm(cur_pos[i])

            if cur_dist > max_dist_from_median:
                max_dist_from_median = cur_dist

        # compute the tps spline
        xs = [pt[0] for pt in cur_pos]
        ys = [pt[1] for pt in cur_pos]

        tps = Tps(xs, ys, self.depths)
        # tps = Tps(xs, ys, depths_curpos)

        # polarize
        phis = np.linspace(-np.pi, np.pi, self.res)
        # rhos = np.linspace(0, max_dist_from_median*1.2 , self.res)
        rhos = np.linspace(0, 1, self.res)
        # rhos = np.linspace(0, 1, 2)

        PI, RI = np.meshgrid(phis, rhos)
        ZI_pol = misc.get_depths_for_polar_grid(PI,RI,tps)



        # # monotonize polarized spline
        monofit = monocol.MonotonicityColwise(RI, ZI_pol)
        ZI_pol_mono = monofit(bw=self.bw)
        # ZI_pol_mono = ZI_pol # to skip monotonization



        # code for drawing polar plots if needed
        if iter is not None:
            plt.figure()
            # plt.axis('equal')
            ax = plt.gca()
            plt.title('Mono regression in polar coordinates ')
            plt.pcolor(PI, RI, ZI_pol_mono, cmap=cm.jet, vmin=np.amin(ZI_pol_mono), vmax=np.amax(ZI_pol_mono))
            plt.colorbar()

            pols = map(misc.cart2pol,xs,ys)
            rs,ps = zip(*pols)

            matplotlib.rcParams.update({'font.size': 7})
            plt.scatter(ps,rs, 30, self.depths, cmap=cm.jet, vmin=np.amin(ZI_pol_mono), vmax=np.amax(ZI_pol_mono))
            for i2 in range(len(xs)):
                ax.annotate(str(i2), (ps[i2],rs[i2]))

            # CS = plt.contour(PI, RI, ZI_pol, levels=[0,3,5,10], colors='k')
            # plt.clabel(CS, inline=1, fontsize=10)
            plt.savefig('output_tsvs/polar_'+format(iter, '04')+'.png', bbox_inches='tight')
            plt.close()




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



        # test code for the presentation figure
        # X_flat = 100+X_flat*100
        # Y_flat = 100+Y_flat*100
        # dprint(min(Y_flat),max(Y_flat))
        # for ii in X_flat:
        #
        #     for j in Y_flat:
        #
        #         XI2[int(math.floor(ii)),int(math.floor(j))] = 0
        # plt.spy(XI2)
        # plt.show()
        # exit(0)

        ZI_pol_mono_flat = ZI_pol_mono.flatten()
        ZI_pol_mono_flat = ZI_pol_mono_flat.reshape(np.shape(X_flat))

        ZI_car_mono = griddata(np.hstack((X_flat,Y_flat)), ZI_pol_mono_flat, (XI2, YI2), method='cubic')
        ZI_car_mono = ZI_car_mono.reshape((self.res,self.res))
        ZIcm = np.nan_to_num(ZI_car_mono) # just making the name shorter for convinience.

        # ZIcm = tps(XI2,YI2) ##for presentation figure on monotonization, code starts

        # ZIcm_max = np.amax(ZIcm)
        # ZIcm_min = np.amin(ZIcm)
        # p = {
        #         'res_ray':25,#50
        #         'dir_ray':60 # 12# number of ray directions
        #     }
        # center = [0.5,0.5]
        # ## GENERATE RAY POINTS, also a 2d array
        # depths_hull_rays = np.zeros((p['dir_ray'],p['res_ray']))
        # depths_ellipse_rays = np.zeros((p['dir_ray'],p['res_ray']))
        #
        # plt.figure()
        # ax = plt.gca()
        # ax.set_xlim(-0.2,1.2)
        # ax.set_ylim(-0.2,1.2)
        # plt.axis('equal')
        # xray = np.zeros_like(depths_hull_rays)
        # yray = np.zeros_like(depths_hull_rays)
        # dprint(np.shape(xray))
        # ax.set_title('Direction of rays')
        #
        #
        # for i in range(p['dir_ray']):
        #
        #     costheta = np.cos(i*2*np.pi/p['dir_ray'])
        #     sintheta = np.sin(i*2*np.pi/p['dir_ray'])
        #     xray[i,:] = center[0]+np.linspace(0,0.5, p['res_ray'])*costheta
        #     yray[i,:] = center[1]+np.linspace(0,0.5, p['res_ray'])*sintheta
        #
        #     indx = map(lambda x: int(math.floor(200*(x)))-1, xray[i,:])
        #     indy = map(lambda x: int(math.floor(200*(x)))-1, yray[i,:])
        #     ZIs = ZIcm[indx,indy]
        #
        #
        #     dprint(np.amax(xray),np.amax(yray),np.amin(xray),np.amin(yray))
        #     ax.scatter(yray[i,:], xray[i,:], 30, ZIs, cmap=cm.YlGnBu, vmin=ZIcm_min, vmax=ZIcm_max, edgecolors='none')
        #
        #
        #
        # plt.show()
        # exit(0)
        # presentation fig code ends


        return cur_pos,XI2,YI2,ZIcm,depths_curpos




    def compute_depth_forces_from_spline(self,cur_pos,XI2,YI2,ZIcm, depths_curpos, iter):
        """
        Args:
            cur_pos:
            res:
            XI2:
            YI2:
            ZIcm:
            depths_curpos: for getting more accuracy instead of computing this from Zcm
        Returns:

        """
        # dprint(self.depths)
        # get point inds on grid
        cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res,XI2,YI2)

        # compute gradients at the point positions in the cur_ids using central differences
        gd = []
        dpfs = []
        alphas = np.zeros(len(cur_ids))
        diffs = np.zeros(len(cur_ids))
        # dprint(depths_curpos)


        for j in range(len(cur_ids)):
        # j = iter%len(cur_pos)

            st_gd = [np.array([0,0])] * len(cur_ids) # stochastic gradient descent gradient that blocks out all others

            ix = cur_ids[j][1];iy = cur_ids[j][0] # swapped ix and iy to make consistent with np indexing/fetching
            gy = (ZIcm[ix+1,iy] - ZIcm[ix-1,iy])/self.twice_grid_side
            gx = (ZIcm[ix,iy+1] - ZIcm[ix,iy-1])/self.twice_grid_side
            gd.append(np.array([gx,gy]))

            st_gd[j] = -np.array([gx,gy])
            # alphas[j] = 0.0333#self.depth_linesearch(ZIcm, XI2, YI2, st_gd, cur_pos,rho=0.5, c=0.0001)
            # alphas[j] = self.depth_linesearch(ZIcm, XI2, YI2, st_gd, cur_pos,rho=0.5, c=0.0001)

            # compute the forces based on the diff between the depths, and the gradients
            diffs[j] = ZIcm[ix,iy]-self.depths[j]
            # diffs[j] = depths_curpos[j]-self.depths[j]
            # dpfs.append(alp*diff*(1/(np.linalg.norm(gd[j])))*gd[j])
            # dpfs.append(alphas[j]*2*diff*gd[j])


        gd = np.array(gd)
        alpha = self.depth_linesearch(ZIcm, XI2, YI2, gd, cur_pos,rho=0.5, c=0.0001)
        alphas = alphas+alpha


        dpfs = np.multiply(gd, alphas[:,np.newaxis])
        dpfs = 2*np.multiply(dpfs, diffs[:,np.newaxis])

        return dpfs



    def get_total_energy(self, ZIcm, XI2, YI2, cur_pos):
        """Computes and returns the total energy.

        Args:
            ZIcm:
            XI2:
            YI2:
            cur_pos:

        Returns:

        """
        N = len(cur_pos)

        cur_dist_mat = self.list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
        dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
        dis_energy /= N**2

        cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res, XI2, YI2)
        depths_curpos = np.zeros(len(cur_ids))

        for i in range(len(cur_ids)):
            depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]

        depth_energy = np.sum((self.depths - depths_curpos)**2)
        depth_energy /= N

        return dis_energy+depth_energy



    def total_linesearch(self, ZIcm, XI2, YI2, gd, cur_pos, rho, c):
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
        fk = self.get_total_engergy(ZIcm, XI2, YI2, xx)
        x = copy.deepcopy(xx)

        while np.linalg.norm(alphak*gk) > self.max_step:
            alphak = alphak*rho
        x = x + alphak*gk

        # compute updated x with original alpha
        fk1 = self.get_total_engergy(ZIcm, XI2, YI2, x)
        # dprint(fk1, fk - c*alphak*np.sum(np.multiply(gk,gk)), gk)
        # while undesirable condition exists
        while fk1 > fk + c*alphak*np.sum(np.multiply(gk,gk)):
            # compute the newer alpha
            alphak = alphak*rho
            # compute the newer x using the new alpha
            x = xx + alphak*gk
            # compute the new objective function value at x
            fk1 = self.get_total_engergy(ZIcm, XI2, YI2, x)
            # dprint(fk1, fk, c*alphak*np.sum(np.multiply(gk,gk)), alphak)
        # exit(0)
        # dprint(alphak)
        return alphak

    def get_total_forces(self, mds_dist_mat, cur_dist_mat, cur_pos, XI2, YI2, ZIcm,fixed_stepsize, depth_factor, smooth_factor, itr):
        """Computes the total force for the next iteration based on the total energy and
        adaptive stepsize.

        Returns:

        """

        # get dist forces gradient
        N = len(cur_pos)

        W = np.reciprocal(mds_dist_mat)
        W = W*W



        np.fill_diagonal(W, 0)
        Z = np.array(cur_pos)
        V = -1*np.ones((N,N))
        np.fill_diagonal(V, N-1)
        B = self.get_B_matrix(N,mds_dist_mat, cur_dist_mat, W)
        V = self.get_V_matrix(mds_dist_mat, W)
        grad_sig = 2*(np.dot(V,Z)-np.dot(B,Z))
        # grad_sig /= 2*N

        grad_sig /= N**2

        # get depth forces gradient
        cur_ids = ipos.get_pos_on_grid2(cur_pos,self.res,XI2,YI2)

        # compute gradients at the point positions in the cur_ids using central differences
        gd = []
        diffs = np.zeros(len(cur_ids))
        for j in range(len(cur_ids)):
            ix = cur_ids[j][1];iy = cur_ids[j][0] # swapped ix and iy to make consistent with np indexing/fetching
            gy = (ZIcm[ix+1,iy] - ZIcm[ix-1,iy])/self.twice_grid_side
            gx = (ZIcm[ix,iy+1] - ZIcm[ix,iy-1])/self.twice_grid_side
            gd.append(np.array([gx,gy]))
            diffs[j] = ZIcm[ix,iy]-self.depths[j]

        gd = np.array(gd)
        grad_dep = 2*np.multiply(gd, diffs[:,np.newaxis])
        grad_dep /= N

        lamda = float(itr)/self.N_i
        lamda = lamda**0.4

        # if (iter > 300):
        #     W3 = W2
        # else:
        #     W3 = (1-lamda)*W + lamda*W2

        # get smoothing forces
        #  https://textons.wordpress.com/2012/10/29/laplacian-regularization-and-trace/
        # gs = []
        # for i in range(N):
        #     ix = cur_ids[i][1];
        #     iy = cur_ids[i][0];
        #     gx = 0
        #     gy = 0
        #     for j in range(N):
        #         jx = cur_ids[j][1];
        #         jy = cur_ids[j][0];
        #         gx += self.A[i,j]*(ix-jx)
        #         gy += self.A[i,j]*(iy-jy)
        #
        #     gs.append(np.array([gx,gy]))
        #
        # grad_sm = np.array(gs)
        # dprint(grad_dep[:5])

        gs = self.A*cur_pos
        grad_sm = np.array(gs)

        # grad_sig = np.zeros_like(grad_sig)

        gd_total = lamda*grad_sig + depth_factor*grad_dep+ smooth_factor*grad_sm
        dprint(itr, float(itr)/self.N_i, (1-lamda),lamda)

        # compute the step using alpha and the total gradient
        if fixed_stepsize is not None:
            alpha = fixed_stepsize

        # else:
        #     if self.adaptive_enabled:
        #         alpha = self.depth_linesearch(ZIcm, XI2, YI2, gd_total, cur_pos,rho=0.5, c=0.0001)
        #         # alpha = 1
        #     else:
        #         alpha = 1
        #         # alpha = self.depth_linesearch(ZIcm, XI2, YI2, gd_total, cur_pos,rho=0.5, c=0.0001)


        # return the step
        tfs = alpha*gd_total
        return tfs, grad_dep, grad_sig, grad_sm


    def get_mds_dc(self, original_points, cur_pos, show_fig, spline_lag, depth_factor, smooth_factor, orig_dist_mat=None,fixed_stepsize=None, names = None):
        """t

        Args:
            points2d_mds: Positions after being projected to 2d.
            alpha: For determining the bands and the outliers.
            N_i: Number of iterations.

        Returns:

        """
        N = len(cur_pos)

        forces = [[],[],[]]

        # get the energies at the begining.
        cur_pos,XI2,YI2,ZIcm, depths_curpos = self.compute_depth_spline(cur_pos)
        self.old_ZIcm = ZIcm
        depth_energy = self.get_depth_engergy(ZIcm, XI2, YI2, cur_pos, depth_factor)
        depth_energy /= N
        forces[0].append(depth_energy)

        if original_points is not None:
            self.original_dist_mat = self.list_to_dist_mat(original_points)
        else:
            self.original_dist_mat = orig_dist_mat

        cur_dist_mat = self.list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
        dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)


        dis_energy /= N**2
        forces[1].append(dis_energy)

        forces[2].append(dis_energy+depth_energy)


        if self.alpha is not None:
            median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(np.array(self.depths), alpha=self.alpha)

        data_dict = {}
        data_dict['orig_depths'] = list(self.depths)
        data_dict['max_orig_depth'] = np.amax(self.depths)
        data_dict['min_orig_depth'] = np.amin(self.depths)
        data_dict['spline_lag'] = spline_lag
        data_dict['N_i'] = self.N_i
        if self.alpha is not None:
            data_dict['median'] = [median]
            data_dict['band50'] = list(band50)
            data_dict['band100'] = list(band100)
            data_dict['outliers'] = list(outliers)
        if names is not None:
            data_dict['names'] = names

        with open("output_tsvs/orig_depths.json", 'w') as outfile:
            json.dump(data_dict, outfile, sort_keys=True, indent=4)

        for iter in range(self.N_i):

            # performing the unconstrained step.
            cur_dist_mat = self.list_to_dist_mat(cur_pos)



            if iter%spline_lag==0:
                cur_pos,XI2,YI2,ZIcm,depths_curpos = self.compute_depth_spline(cur_pos, iter)
                ZIcm = (ZIcm + self.old_ZIcm)/2
                self.old_ZIcm = ZIcm

                self.adaptive_enabled = False

            tfs,dfgd,dgd, sgd = self.get_total_forces(self.original_dist_mat, cur_dist_mat,cur_pos,XI2,YI2, ZIcm,fixed_stepsize, depth_factor, smooth_factor, iter)


            #vis
            xs = [pos[0] for j,pos in enumerate(cur_pos)] # after centering the cur_pos about median
            ys = [pos[1] for j,pos in enumerate(cur_pos)]

            ZIcm_max = np.amax(ZIcm)
            ZIcm_min = np.amin(ZIcm)
            if iter%spline_lag==0:

                ax_spline = plt.subplot(2,3,1)
                plt.figure()
                plt.axis('equal')
                plt.title('Monotonized TPS')
                plt.pcolor(XI2, YI2, ZIcm, cmap=cm.YlGnBu, vmin=ZIcm_min, vmax=ZIcm_max)
                # dprint('iter',iter, ZIcm_max,ZIcm_min) #-0.18, 1
                # plt.pcolor(XI2, YI2, ZIcm, cmap=cm.YlGnBu, vmin=ZIcm_min, vmax=ZIcm_max)

                plt.scatter(xs, ys, 30, self.depths, cmap=cm.YlGnBu, vmin=ZIcm_min, vmax=ZIcm_max)
                plt.colorbar()
                CS = plt.contour(XI2, YI2, ZIcm, levels=[1,2,4], colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                ax = plt.gca()
                for i2 in range(len(xs)):
                    ax.annotate(str(i2), (xs[i2],ys[i2]))
                imname = 'output_tsvs/spline_'+format(iter, '04')+'.png'
                plt.savefig(imname, dpi=100)
                plt.close()
                im = Image.open(imname)
                im.putalpha(192)
                im.save(imname)

                fig = plt.figure(frameon=False)
                fig.set_size_inches(w=4,h=4)

                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.axis('equal')
                plt.pcolor(XI2, YI2, ZIcm, cmap=cm.YlGnBu, vmin=ZIcm_min, vmax=ZIcm_max)
                levels = np.arange(0, 1, 0.1)
                # CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.1, levels=levels, colors='k',
                                 antialiased=True)
                # plt.clabel(CS, inline=1, fontsize=8)
                plt.axis('off')

                imname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                fig.savefig(imname, dpi=100)
                im = Image.open(imname)
                im.putalpha(192)
                # im.putalpha(255)
                im.save(imname)
                plt.close()

                # # for printing out the colormap color values
                # cmap = cm.autumn_r
                # for ic in range(cmap.N):
                #     rgb = cmap(ic)[:3] # will return rgba, we take only first 3 so we get rgb
                #     print(ic,matplotlib.colors.rgb2hex(rgb))
                # dprint(cmap.N)
                # exit(0)


            norms = np.linalg.norm(tfs, axis=1)
            pos = np.where(norms>self.max_step)
            tfs[pos] = (tfs[pos]/norms[pos,None])*self.max_step

            for j in range(len(cur_pos)):
                cur_pos[j] = cur_pos[j] - tfs[j]



            cur_dist_mat = self.list_to_dist_mat(cur_pos)
            dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
            dis_energy /= N**2
            forces[1].append(dis_energy)

            depth_energy = self.get_depth_engergy(ZIcm, XI2, YI2,  cur_pos, depth_factor)
            depth_energy /= N
            forces[0].append(depth_energy) # just appending the previously computed depth energy to avoid exception if dis_force throws
            # something far away
            forces[2].append(dis_energy+depth_energy)
            # dprint('dis_energy,dep_energy i  ', iter, dis_energy, depth_energy)


            xs_new = [pos[0] for j,pos in enumerate(cur_pos)]
            ys_new = [pos[1] for j,pos in enumerate(cur_pos)]

            if iter%spline_lag==0:

                ax_dis = plt.subplot(2,3,4)
                plt.figure()
                # plt.axis('equal')
                plt.title('Stress (dist) energy')
                plt.plot(range(2+(iter)),forces[1],'r--')
                plt.legend(loc=2)
                ax = plt.gca()
                maxe0 = max(forces[1])
                ax.set_ylim([-maxe0*0.1,maxe0*1.1])
                plt.savefig('output_tsvs/dis_'+format(iter, '04')+'.png')
                plt.close()

                ax_dep = plt.subplot(2,3,5)
                plt.figure()

                # plt.axis('equal')
                plt.title('Depth energy')
                plt.plot(range(2+(iter)),forces[0],'g--')
                plt.legend(loc=2)
                ax = plt.gca()
                maxe1 = max(forces[0])
                ax.set_ylim([-maxe1*0.1,maxe1*1.1])
                plt.savefig('output_tsvs/dep_'+format(iter, '04'))
                plt.close()

                total_energy = [x + y for x, y in zip(forces[0], forces[1])]
                # ax_tot = plt.subplot(2,3,6)
                plt.figure()
                ax = plt.gca()
                # plt.axis('equal')
                plt.title('Total energy')
                # plt.plot(range(2+(iter)),forces[1],'r--',label="MDS stress")
                lns1 = ax.plot(range(2+(iter)),forces[0],'g--',label="Penalty")
                lns2 = ax.plot(range(2+(iter)),total_energy,'b--',label="Total Energy")
                plt.legend(loc=1,prop={'size':14})
                plt.xlabel('Iterations', fontsize=14)
                plt.ylabel('Energy (Penalty/Total)',fontsize=14)
                plt.grid(True)

                ax2 = ax.twinx()
                lns3 = ax2.plot(range(2+(iter)),forces[1],'r--',label="MDS stress")
                ax2.set_ylim([0.008,0.010])
                ax2.set_ylabel('Energy (MDS)', fontsize=14)
                # plt.legend(loc=1,prop={'size':14})

                # added these three lines
                lns = lns1+lns3+lns2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=1,prop={'size':14})

                maxe2 = max(forces[2])
                # ax.set_ylim([-maxe2*0.1,maxe2*1.1])
                ax.set_ylim([0,maxe2*1.1])
                ax.tick_params(labelsize=14)
                ax2.tick_params(labelsize=14)
                plt.savefig('output_tsvs/tot_'+format(iter, '04')+'.png',bbox_inches='tight')
                plt.close()

                if iter == self.N_i - 1:
                    np.save('output_tsvs/forces_'+str(iter)+'.npy', forces)

            # showFig = False
            if show_fig:
                plt.show(block=False)
            else:
                # plt.savefig('output/'+format(iter, '04')+'.png', bbox_inches='tight')

                with open('output_tsvs/'+format(iter, '04')+'.tsv', 'wb') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    header = ['cpx', 'cpy', 'npx', 'npy', 'dpfx', 'dpfy', 'dfx', 'dfy','oridep']
                    writer.writerow(header)
                    for i in range(N):
                        row = [round(xs[i],4), round(ys[i],4), round(xs_new[i],4), round(ys_new[i],4),
                               round(dfgd[i,0],4), round(dfgd[i,1],4), round(dgd[i,0],4), round(dgd[i,1],4),
                               round(self.depths[i],4)]
                        writer.writerow(row)

                plt.close()

        return cur_pos

    def get_rmds_pos(self, original_points, cur_pos, show_fig, spline_lag, depth_factor, orig_dist_mat=None,fixed_stepsize=None, names = None):

        dprint("hello2")
        pass