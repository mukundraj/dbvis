'''
Created on Dec 29, 2016

Class to perform the depth constrained MDS.

@author: mukundraj
'''
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

class dc_mds(object):
    """

    """

    def __init__(self, depths, N_i, M_i,res,bw, max_step, num_old, num_old_spline, depth_type, alpha=None):
        """

        Args:
            depths:

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
        self.running_avg = []
        for i in range(len(depths)):
            self.running_avg.append(np.array([0.0,0.0]))
        # [np.array([0.0,0.0])] * len(depths)

        self.adaptive_enabled = False
        self.running_avg_enabled = False

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

    def get_depth_engergy(self,depths, ZIcm, XI2, YI2, cur_pos, res):
        """Get the curre

        Args:
            ZIcm (2d array):
            cur_ids (list):

        Returns:
            depth_energy (float):

        """
        cur_ids = ipos.get_pos_on_grid2(cur_pos,res, XI2, YI2)
        depths_curpos = np.zeros(len(cur_ids))
        # distance.euclidean(cur_pos[],pos[j])
        for i in range(len(cur_ids)):
            depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]




        depth_energy = np.sqrt(np.sum((depths - depths_curpos)**2))

        return depth_energy

    def get_V_matrix(self, mds_dist_mat):
        """ Get V matrix considering weights. This is for
        reducing the impact of distant forces.

        Args:
            mds_dist_mat:

        Returns:

        """

        m,N = np.shape(mds_dist_mat)

        # V = -copy.deepcopy(W)
        # for i in range(N):
        #     V[i,i] = -np.sum(V[i,:])

        V = np.ones((N,N))
        V *= -1
        np.fill_diagonal(V,N-1)

        return V

    def get_B_matrix(self,N, mds_dist_mat, cur_dist_mat):
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
                    # B[i,j] = -W[i,j]*mds_dist_mat[i,j]/cur_dist_mat[i,j]
                    B[i,j] = -mds_dist_mat[i,j]/cur_dist_mat[i,j]
                elif i!=j:
                    B[i,j] = 0

        for i in range(N):
            B[i,i] = -(np.sum(B[i,:]) - B[i,i])


        return B

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

        tps = Tps(xs, ys, depths_curpos)
        # polarize
        phis = np.linspace(-np.pi, np.pi, self.res)
        # rhos = np.linspace(0, max_dist_from_median*1.2 , self.res)
        rhos = np.linspace(0, 1, self.res)

        PI, RI = np.meshgrid(phis, rhos)
        ZI_pol = misc.get_depths_for_polar_grid(PI,RI,tps)



        # # monotonize polarized spline
        # monofit = monocol.MonotonicityColwise(RI, ZI_pol)
        # ZI_pol_mono = monofit(bw=self.bw)
        ZI_pol_mono = ZI_pol # to skip monotonization

        # code for drawing polar plots if needed
        # if iter is not None:
        #     plt.figure()
        #     # plt.axis('equal')
        #     ax = plt.gca()
        #     plt.title('Mono regression in polar coordinates ')
        #     plt.pcolor(PI, RI, ZI_pol_mono, cmap=cm.jet, vmin=np.amin(ZI_pol_mono), vmax=np.amax(ZI_pol_mono))
        #     plt.colorbar()
        #
        #     pols = map(misc.cart2pol,xs,ys)
        #     rs,ps = zip(*pols)
        #
        #     matplotlib.rcParams.update({'font.size': 7})
        #     plt.scatter(ps,rs, 30, self.depths, cmap=cm.jet, vmin=np.amin(ZI_pol_mono), vmax=np.amax(ZI_pol_mono))
        #     for i2 in range(len(xs)):
        #         ax.annotate(str(i2), (ps[i2],rs[i2]))
        #
        #     # CS = plt.contour(PI, RI, ZI_pol, levels=[0,3,5,10], colors='k')
        #     # plt.clabel(CS, inline=1, fontsize=10)
        #     plt.savefig('output/'+format(iter, '04')+'.png', bbox_inches='tight')
        #     plt.close()



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
        # dprint(np.shape(ZI_pol_mono_flat), np.shape(X_flat), np.shape(Y_flat))
        # exit(0)

        ZI_car_mono = griddata(np.hstack((X_flat,Y_flat)), ZI_pol_mono_flat, (XI2, YI2), method='cubic')
        ZI_car_mono = ZI_car_mono.reshape((self.res,self.res))
        ZIcm = np.nan_to_num(ZI_car_mono) # just making the name shorter for convinience.





        return cur_pos,XI2,YI2,ZIcm,depths_curpos


    def depth_linesearch(self, ZIcm, XI2, YI2, gd, cur_pos, rho, c):
        """

        Args:
            f:
            d:
            x:
            rho:
            c:

        Returns:

        """
        alphak=1
        # compute fk and gk
        gk = np.array(gd)


        # store orig x
        xx = np.array(cur_pos)
        fk = self.get_depth_engergy(ZIcm, XI2, YI2, xx)
        x = copy.deepcopy(xx)

        while np.linalg.norm(alphak*gk) > self.max_step:
            alphak = alphak*rho
        x = x - alphak*gk

        # compute updated x with original alpha
        fk1 = self.get_depth_engergy(ZIcm, XI2, YI2, x)
        # dprint(fk1, fk - c*alphak*np.sum(np.multiply(gk,gk)), gk)
        # while undesirable condition exists
        while fk1 > fk - c*alphak*np.sum(np.multiply(gk,gk)):
            # compute the newer alpha
            alphak = alphak*rho
            # compute the newer x using the new alpha
            x = xx - alphak*gk
            # compute the new objective function value at x
            fk1 = self.get_depth_engergy(ZIcm, XI2, YI2, x)
            # dprint(fk1, fk, c*alphak*np.sum(np.multiply(gk,gk)), alphak)
        # exit(0)
        # dprint(alphak)
        return alphak

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

        np.fill_diagonal(W, 0)

        Z = np.array(cur_pos)
        V = -1*np.ones((N,N))
        np.fill_diagonal(V, N-1)

        B = self.get_B_matrix(N,mds_dist_mat, cur_dist_mat, W)
        # dprint(B)
        # exit(0)

        # dmat = np.copy(mds_dist_mat)
        # dmat[dmat==0]=1
        # V = np.divide(V,dmat)
        # B = np.divide(B,dmat)

        # W = np.copy(mds_dist_mat)


        V = self.get_V_matrix(mds_dist_mat,W)
        # dprint(V)
        # exit(0)

        grad_sig = 2*(np.dot(V,Z)-np.dot(B,Z))


        # Zkpo = (1/float(N))*np.dot(B,Z)



        # dprint(np.shape(Zkpo))

        # dprint(Zkpo)
        # exit(0)

        SCAMOF_FAC = float(1)/(2*len(cur_pos)) # denominator for scamof gradient descent
        dfs = SCAMOF_FAC*grad_sig

        return dfs

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

    def get_total_forces(self, mds_dist_mat, cur_dist_mat, cur_pos, XI2, YI2, ZIcm,fixed_stepsize, depth_factor, iter):
        """Computes the total force for the next iteration based on the total energy and
        adaptive stepsize.

        Returns:

        """

        # get dist forces gradient
        N = len(cur_pos)

        # W = np.reciprocal(mds_dist_mat)

        # np.fill_diagonal(W, 0)
        Z = np.array(cur_pos)
        V = -1*np.ones((N,N))
        np.fill_diagonal(V, N-1)
        B = self.get_B_matrix(N,mds_dist_mat, cur_dist_mat)
        V = self.get_V_matrix(mds_dist_mat)
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

        # grad_fac = min(10,1+math.floor(iter/50))
        # dprint(grad_fac)

        gd_total = grad_sig+depth_factor*grad_dep


        # compute the step using alpha and the total gradient
        if fixed_stepsize is not None:
            alpha = fixed_stepsize

        else:
            if self.adaptive_enabled:
                alpha = self.depth_linesearch(ZIcm, XI2, YI2, gd_total, cur_pos,rho=0.5, c=0.0001)
                # alpha = 1
            else:
                alpha = 1
                # alpha = self.depth_linesearch(ZIcm, XI2, YI2, gd_total, cur_pos,rho=0.5, c=0.0001)


        # return the step
        tfs = alpha*gd_total
        return tfs, grad_dep, grad_sig


    def cur_pos_setter(self, cur_pos, iter):
        """Sets the old_poses and running average (running avg assumed to have been initialized before)

        Args:
            cur_pos:
            j:

        Returns:

        """

        j = iter%self.num_old
        # prev_j = (j+1)%self.num_old

        for i in range(len(cur_pos)):

            # self.running_avg[i][0] -= self.old_poses[i,j*2]
            # self.running_avg[i][1] -= self.old_poses[i,j*2+1]

            self.old_poses[i,j] = cur_pos[i][0]
            self.old_poses[i,j+self.num_old] = cur_pos[i][1]

            # self.running_avg[i][0] += self.old_poses[i,j*2]
            # self.running_avg[i][1] += self.old_poses[i,j*2+1]

        # dprint(np.mean(self.old_poses[i,0:self.num_old]))
            self.running_avg[i][0] = np.mean(self.old_poses[i,0:self.num_old])
            self.running_avg[i][1] = np.mean(self.old_poses[i,self.num_old:])

        # dprint(self.old_poses[0,:])
        # dprint(self.running_avg[0])



    def center_about_median(self, cur_pos):
        """

        Args:
            cur_pos:

        Returns:

        """

         # compute depths based on cur_pos


        # analyser = hsa.analyzer(members=cur_pos, grid_members=None, kernel=None) # gridpoints2d is vestigial here
        # depths_curpos = analyser.get_depths_extern()

        # G = linear_kernel(cur_pos)
        # analyser_spa = spa.SpatialDepth(G)
        # depths_curpos = analyser_spa.get_depths_from_gram()

        # No need to scale here as depth only used to determine the center?
        # depths_curpos = self.depths_scaler(depths_curpos, orig_depths_scale, orig_depths_min) # scaling the range of depth_curpos to match range of depth

        # sorted_inds = sorted(range(depths_curpos.shape[0]), key=lambda k: depths_curpos[k])
        # sorted_inds = sorted(range(ori_depths.shape[0]), key=lambda k: ori_depths[k])
        median_pos = copy.deepcopy(cur_pos[self.ori_median_ind])

        dprint(self.ori_median_ind)

        for i in range(len(cur_pos)):
            cur_pos[i] -= median_pos

        return cur_pos


    def histmatch(self, ta1, ta2, bw=0.1, res=1000, showplot=False):
        """Matches the histogram of a1 to histogram on a2.

        Args:
            a1: original input historgram
            a2: the target histogram to be matched to
            bw: bandwidth
            res: the resolution for computing the cdf
        Returns:

        """

        xs = np.linspace(0,1,res)
        n = len(ta1)

        a1 = copy.deepcopy(ta1).reshape((n,1))
        a2 = copy.deepcopy(ta2).reshape((n,1))

        # dprint(np.shape(a1,),np.shape(a2))
        # get T(r)
        kde_a1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a1)
        a1s = np.exp(kde_a1.score_samples(xs.reshape((res,1))))
        a1s = a1s/np.sum(a1s) # normalize for the grid size

        # hist, edges = np.histogram(self.depths,bins=20)

        # hist, edges = np.histogram(depths_curpos,bins=20)
        # plt.plot(hist, color='g', label='2D')
        #

        #
        # hist, edges = np.histogram(self.depths,bins=20)


        Tr = np.cumsum(a1s)


        # get G(z)
        kde_a2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a2)
        a2s = np.exp(kde_a2.score_samples(xs.reshape((res,1))))
        a2s = a2s/np.sum(a2s) # normalize for the grid size
        Gz = np.cumsum(a2s)



        # get G^-1(z)
        Gz_inv = interpolate.interp1d(Gz, xs, fill_value="extrapolate")

        # do the conversion and return the matched histogram
        matched = []
        for i,d in enumerate(a1):
            id = int(math.floor(d*(res-1)))
            # dprint(d,id)
            matched.append(Gz_inv(Tr[id]))

        matched = np.array(matched)

        if showplot==True:

            cdf = a1s.cumsum()
            cdf = 1 * cdf / cdf[-1] # normalize
            # use linear interpolation of cdf to find new pixel values
            dep_eq = np.interp(ta1, xs, cdf)
            dep_eq = self.depths_scaler(dep_eq)

            kde_eq = KernelDensity(kernel='gaussian', bandwidth=bw).fit(copy.deepcopy(dep_eq).reshape((n,1)))
            eq_s = np.exp(kde_eq.score_samples(xs.reshape((res,1))))
            eq_s = eq_s/np.sum(eq_s)


            kde_matched = KernelDensity(kernel='gaussian', bandwidth=bw).fit(copy.deepcopy(matched).reshape((n,1)))
            matched_s = np.exp(kde_matched.score_samples(xs.reshape((res,1))))
            matched_s = matched_s/np.sum(matched_s) # normalize for the grid size
            plt.plot(xs, a1s, color='r', label='original')
            plt.plot(xs, a2s, color='g', label='2D')
            plt.plot(xs, matched_s, color='b', label='matched')
            plt.plot(xs, eq_s, color='y', label='equalized')
            plt.legend(loc='best')
            plt.show()
            exit(0)

        # dprint(np.shape(ta1), np.shape(matched))
        # exit(0)

        return matched


    def get_mds_dc(self, original_points, cur_pos, show_fig, spline_lag, depth_factor, orig_dist_mat=None,fixed_stepsize=None, names = None):
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
        # self.running_avg = copy.deepcopy(cur_pos)
        #dprint(self.running_avg)


        for j in range(self.num_old):
            self.cur_pos_setter(cur_pos,j)

        for j in range(self.num_old_spline):
            self.old_ZIcm_s[:,:,j] = ZIcm



        iter_at_last_spline_compute = 0

        if self.alpha is not None:
            median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(np.array(self.depths), alpha=self.alpha)

        data_dict = {}
        data_dict['orig_depths'] = list(self.depths)
        data_dict['max_orig_depth'] = np.amax(self.depths)
        data_dict['min_orig_depth'] = np.amin(self.depths)
        data_dict['spline_lag'] = spline_lag
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

            # dis_forces = self.get_dist_forces(mds_dist_mat=self.original_dist_mat, cur_dist_mat=cur_dist_mat, cur_pos=cur_pos, k=1)
            # dfs = np.array(dis_forces)

            if iter%spline_lag==0:
                cur_pos,XI2,YI2,ZIcm,depths_curpos = self.compute_depth_spline(cur_pos, iter)
                ZIcm = (ZIcm + self.old_ZIcm)/2
                self.old_ZIcm = ZIcm

                # self.old_ZIcm_s[:,:,spline_id%self.num_old_spline] = ZIcm
                # spline_id += 1
                # ZIcm = np.sum(self.old_ZIcm_s,2)

                self.adaptive_enabled = False
                iter_at_last_spline_compute = iter

            # output
            # dpfs = self.compute_depth_forces_from_spline(cur_pos,XI2,YI2,ZIcm,depths_curpos, iter)
            # dprint(max(depths_curpos), max(self.depths)).

            # tfs,dfgd,dgd = self.get_total_forces(self.original_dist_mat, cur_dist_mat, self.running_avg,XI2,YI2, ZIcm)

            tfs,dfgd,dgd = self.get_total_forces(self.original_dist_mat, cur_dist_mat,cur_pos,XI2,YI2, ZIcm,fixed_stepsize, depth_factor, iter)

            # self.cur_pos_setter(tfs,iter)
            # dprint('running avg sum', np.sum(self.running_avg))

            #vis
            xs = [pos[0] for j,pos in enumerate(cur_pos)] # after centering the cur_pos about median
            ys = [pos[1] for j,pos in enumerate(cur_pos)]

            # arrow_scale = 1
            #
            # fig = plt.figure()
            # plt.suptitle('Iteration: '+str(iter))


            ZIcm_max = np.amax(ZIcm)
            ZIcm_min = np.amin(ZIcm)
            if iter%spline_lag==0:

                ax_spline = plt.subplot(2,3,1)
                plt.figure()
                plt.axis('equal')
                plt.title('Monotonized TPS')
                plt.pcolor(XI2, YI2, ZIcm, cmap=cm.jet, vmin=ZIcm_min, vmax=ZIcm_max)
                plt.scatter(xs, ys, 30, self.depths, cmap=cm.jet, vmin=ZIcm_min, vmax=ZIcm_max)
                plt.colorbar()
                CS = plt.contour(XI2, YI2, ZIcm, levels=[1,2,4], colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                ax = plt.gca()
                for i2 in range(len(xs)):
                    ax.annotate(str(i2), (xs[i2],ys[i2]))
                plt.savefig('output_tsvs/spline_'+format(iter, '04')+'.png')
                plt.close()


                fig = plt.figure(frameon=False)
                fig.set_size_inches(w=4,h=4)

                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.axis('equal')
                plt.pcolor(XI2, YI2, ZIcm, cmap=cm.jet, vmin=ZIcm_min, vmax=ZIcm_max)
                CS = plt.contour(XI2, YI2, ZIcm,alpha=1.0, linewidths=0.5, levels=[0.5*(ZIcm_max+ZIcm_min)], colors='k')
                plt.clabel(CS, inline=1, fontsize=8)
                plt.axis('off')
                # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # plt.savefig('output_tsvs/spline_only_'+format(iter, '04')+'.png',bbox_inches='tight', pad_inches=0)
                # ax.imshow(fig, aspect='normal')
                imname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
                fig.savefig(imname, dpi=100)
                im = Image.open(imname)
                im.putalpha(64)
                im.save(imname)
                plt.close()



            # plt.subplot(2,3,2)
            # plt.axis('equal')
            # plt.title('Forces: dis(red), dep(green)')
            # plt.scatter(xs, ys, 30, self.depths, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
            # plt.colorbar()
            # ax = plt.gca()
            # ax.quiver(xs, ys,  -dfgd[:,0], -dfgd[:,1], angles='xy',scale=arrow_scale,color='g')
            # ax.quiver(xs, ys,  -dgd[:,0], -dgd[:,1], angles='xy',scale=arrow_scale,color='r')
            # for i2 in range(len(xs)):
            #     ax.annotate(str(i2), (xs[i2],ys[i2]))

            if iter - iter_at_last_spline_compute == 24:
                self.adaptive_enabled=True


            norms = np.linalg.norm(tfs, axis=1)
            pos = np.where(norms>self.max_step)
            tfs[pos] = (tfs[pos]/norms[pos,None])*self.max_step

            for j in range(len(cur_pos)):
                cur_pos[j] = cur_pos[j] - tfs[j]

            # cur_pos = self.center_about_median(cur_pos)

            cur_dist_mat = self.list_to_dist_mat(cur_pos)
            dis_energy = np.sum(((self.original_dist_mat-cur_dist_mat)/2)**2)
            dis_energy /= N**2
            forces[1].append(dis_energy)

            depth_energy = self.get_depth_engergy(ZIcm, XI2, YI2,  cur_pos, depth_factor)
            depth_energy /= N
            forces[0].append(depth_energy) # just appending the previously computed depth energy to avoid exception if dis_force throws
            # something far away
            forces[2].append(dis_energy+depth_energy)
            dprint('dis_energy,dep_energy i  ', iter, dis_energy, depth_energy)


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
                ax_tot = plt.subplot(2,3,6)
                plt.figure()
                # plt.axis('equal')
                plt.title('Total energy')
                plt.plot(range(2+(iter)),total_energy,'b--')
                plt.legend(loc=2)
                ax = plt.gca()
                maxe2 = max(forces[2])
                ax.set_ylim([-maxe2*0.1,maxe2*1.1])
                plt.savefig('output_tsvs/tot_'+format(iter, '04')+'.png')
                plt.close()

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



# tfs = dfs
# norms = np.linalg.norm(tfs, axis=1)
# pos = np.where(norms>self.max_step)
# tfs[pos] = (tfs[pos]/norms[pos,None])*self.max_step

# cur_dist_mat = self.list_to_dist_mat(cur_pos)
# dis_energy = np.sqrt(np.sum(((mds_dist_mat-cur_dist_mat)/2)**2))
# dprint('dis_energy,dep_energy i M',iter, dis_energy, depth_energy)
#
# forces[1].append(dis_energy)
# depth_energy = self.get_depth_engergy(ZIcm, XI2, YI2,  cur_pos)
# forces[0].append(depth_energy)

# dprint('depth energy', iter, iter_M, depth_energy)