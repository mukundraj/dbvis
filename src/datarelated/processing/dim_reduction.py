'''
Created on Feb 3, 2016

Functions to reduce dimensions of the graph data.

@author: mukundraj
'''

import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn import manifold
import src.visualization.ocmds as oc_mds
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import scipy.spatial.distance as distance
from itertools import combinations
from datarelated.processing.depths import get_median_and_bands
import src.analysis.halfspace_analyzer as hsa
from src.utils.interpolation.Tps import Tps
from libs.productivity import dprint
import src.datarelated.processing.misc as misc
import src.utils.MonotonicityColwise as monocol
from scipy.interpolate import griddata
import src.utils.interpolation.positions as ipos

def get_pca_projections(A_vectorized):
    """ Straightens the matrices and performs PCA analysis

    Args:
        A_vectorized (2D numpy array): A stack of vectors.

    Returns:
        pca_pos (2D numpy array): Positions after projecting the data onto principal
            components.

    """

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    dprint(np.shape(X))
    dprint(np.shape(A_vectorized))

    pca = RandomizedPCA(n_components=2)
    pca_pos = pca.fit_transform(np.transpose(A_vectorized))
    return pca_pos


def get_nmds_projections(data_list, similarities, metric_mds=True):
    """

    Returns:
        points2d (list of 1d array): Points in 2d using the NMDS.

    """

    seed = np.random.RandomState(seed=3)
    # seed = np.random.RandomState(seed=13)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_

    if metric_mds:
        npos = pos # return metric MDS if requested
    else: # else compute and return non-metric mds
        nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                            dissimilarity="precomputed", random_state=seed, n_jobs=1,
                            n_init=1)
        npos = nmds.fit_transform(similarities, init=pos)




    # pos = npos
    # xpos = pos[:,0] - np.amin(pos[:,0])
    # xpos = xpos / np.amax(xpos)
    # ypos = pos[:,1] - np.amin(pos[:,1])
    # ypos = ypos / np.amax(ypos)
    #
    # points2d = []
    # for i in range(len(pos)):
    #     points2d.append(np.array([xpos[i],ypos[i]]))

    npos = npos - np.amin(npos)
    npos = npos / np.amax(npos)
    points2d = []
    for i in range(len(npos)):
        points2d.append((npos[i,:]))
        #points2d.append((functions[i]))

    return points2d

# def get_ocnmds_projections(similarities, depths, alpha):
#     """Order constrained MDS
#
#     Returns:
#
#     """
#
#     seed = np.random.RandomState(seed=3)
#     mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
#                        dissimilarity="precomputed", n_jobs=1)
#     pos = mds.fit(similarities).embedding_
#     ocnmds = oc_mds.CMDS(depths, alpha, n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                             dissimilarity="precomputed", random_state=seed, n_jobs=1,
#                             n_init=1)
#
#     #npos = ocnmds.fit_transform(similarities, init=pos)
#     npos = pos
#
#     npos = npos - np.amin(npos)
#     npos = npos / np.amax(npos)
#
#     points2d = []
#     for i in range(len(npos)):
#         points2d.append((npos[i,:]))
#         #points2d.append((functions[i]))
#
#     return points2d


def list_to_dist_mat(pos):
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



def get_spline_forces_lou(cur_pos, depths, alp=0.01):
    """Leave one out version of spline force.

    Args:
        cur_pos:
        depths:

    Returns:
        sp_forces(list): A list of vectors denoting forces due to the interpolation.
    """

    sp_forces = []

    xs = [pos[0] for j,pos in enumerate(cur_pos)]
    ys = [pos[1] for j,pos in enumerate(cur_pos)]

    for i in range(len(depths)):
        x = [pos[0] for j,pos in enumerate(cur_pos) if j!=i]
        y = [pos[1] for j,pos in enumerate(cur_pos) if j!=i]
        ds = [depth for j,depth in enumerate(depths) if j!=i]
        rbf = Rbf(x, y, ds, function='thin-plate',smooth=0)
        # dprint(i,ds-rbf(x,y))
        # dprint()


        delta = 0.01
        ddelta = 2*delta # double delta
        gx = (rbf(xs[i]+delta,ys[i]) - rbf(xs[i]-delta,ys[i]))/ddelta
        gy = (rbf(xs[i],ys[i]+delta) - rbf(xs[i],ys[i]-delta))/ddelta
        gd = np.array([gx,gy])

        #get diff
        diff = depths[i] - rbf(cur_pos[i][0],cur_pos[i][1])

        force = alp*diff*(1/np.linalg.norm(gd))*gd

        # sp_forces.append(np.array([gx,gy]))
        sp_forces.append(force)


        if i==30:
            dprint(force)
            xi = np.linspace(min(xs), max(xs), 200)
            yi = np.linspace(min(ys), max(ys), 200)
            XI, YI = np.meshgrid(xi, yi)
            ZI = rbf(XI, YI)
            # gy, gx = np.gradient(ZI, .2, .2)
            plt.figure()
            # plt.subplot(2, 2, 1)
            plt.pcolormesh(XI, YI, ZI, cmap=cm.jet)
            plt.scatter(xs, ys, 100, depths, cmap=cm.jet)
            plt.title('RBF interpolation - multiquadrics')
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.colorbar()
            plt.show(block=False)

    # gs=np.array(sp_forces)
    # dprint(gs)
    # # exit(0)
    # plt.subplot(2,2,2)
    # ax = plt.gca()
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(0, 1)
    # ax.quiver(xs, ys,  gs[:,0], gs[:,1])
    #
    # plt.subplot(2,2,3)
    # ax = plt.gca()
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(0, 1)
    # ax.quiver(xs, ys,  gx[::40], gy[::40],linewidth=0.1*np.ones(len(gx[::40])))
    #
    # plt.show()
    # dprint(sp_forces)
    return sp_forces

def get_spline_forces_pp(cur_pos, depths, alp=10.0, vis_vert_id=0):
    """Prior priority version of spline force. Only forces with higher depth members matter.

    Args:
        cur_pos:
        depths:
        alp(int): parameter controlling the step size
        vis_vert_id(int): Id of the vertex for which the spline is to be returned for visualization

    Returns:
        sp_forces(list): A list of vectors denoting forces due to the interpolation.
        vis_vert_spline: The spline corresponding to the vis vertex
    """
    N = len(depths)
    sp_forces = [np.array([0.0,0.0])] * N
    cctr = np.zeros(N) # contribution counter array
    xs = [pos[0] for j,pos in enumerate(cur_pos)]
    ys = [pos[1] for j,pos in enumerate(cur_pos)]

    sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])

    for i in range(N-1):

        # get set of members with depth >= d[i], update the cctr values accordingly
        high_ids = np.where(depths>depths[sorted_inds[i]])
        low_ids = np.where(depths<=depths[sorted_inds[i]])
        high_ids = map(int,high_ids[0])
        low_ids = map(int,low_ids[0])

        if(len(high_ids)>1): # if length 1 then a. can't make tps, b. not much info about the gradients
            dprint(low_ids, high_ids)
            cctr[low_ids] += 1
            # build the rbf using above set
            x = [cur_pos[j][0] for j in high_ids]
            y = [cur_pos[j][1] for j in high_ids]
            ds = [depths[j] for j in high_ids]
            dprint('ds', ds)
            rbf = Rbf(x, y, ds, function='thin-plate')
            if i==vis_vert_id:
                vis_vert_spline = rbf

            # iterate and get forces for other members (non set)
            for j in low_ids:
                #get diff
                diff = rbf(cur_pos[j][0],cur_pos[j][1]) - depths[j]
                dprint(diff)
                #get gradient
                delta = 0.01
                ddelta = 2*delta # double delta
                gx = (rbf(xs[j]+delta,ys[j]) - rbf(xs[j]-delta,ys[j]))/ddelta
                gy = (rbf(xs[j],ys[j]+delta) - rbf(xs[j],ys[j]-delta))/ddelta
                gd = np.array([gx,gy])
                dprint(gd)

                #get and update force
                force = alp*diff*(1/np.linalg.norm(gd))*gd
                sp_forces[j] = sp_forces[j]+force

    # normalize forces by cctr
    for i in range(N):
        if cctr[i] != 0:
            sp_forces[i] = sp_forces[i] / cctr[i]




    return sp_forces, vis_vert_spline

def get_spline_forces_bp(cur_pos, depths, alpha, alp=-0.01):
    """Band priority version of spline force. Only forces from members in higher band matter.

    Args:
        cur_pos:
        depths:
        alp(int): parameter controlling the step size

    Returns:
        sp_forces(list): A list of vectors denoting forces due to the interpolation.
    """
    N = len(depths)
    sp_forces = [np.array([0.0,0.0])] * N
    cctr = np.zeros(N) # contribution counter array
    xs = [pos[0] for j,pos in enumerate(cur_pos)]
    ys = [pos[1] for j,pos in enumerate(cur_pos)]
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths, alpha=alpha)

    all = set(range(N))
    # bands = [[median], band50, band100]
    bands = [band50, band100]

    for band in bands:

        # get set of members with depth >= d[i], update the cctr values accordingly
        high_ids = band
        low_ids = all.difference(high_ids)
        high_ids = list(map(int,high_ids))
        low_ids = list(map(int,low_ids))
        cctr[low_ids] += 1


        # build the rbf using above set
        x = [cur_pos[j][0] for j in high_ids]
        y = [cur_pos[j][1] for j in high_ids]
        ds = [depths[j] for j in high_ids]


        rbf = Rbf(x, y, ds, function='thin-plate')


        # iterate and get forces for other members (non set)
        for j in low_ids:


            #get diff
            diff = rbf(cur_pos[j][0],cur_pos[j][1]) - depths[j]

            #get gradient
            delta = 0.01
            ddelta = 2*delta # double delta
            gx = (rbf(xs[j]+delta,ys[j]) - rbf(xs[j]-delta,ys[j]))/ddelta
            gy = (rbf(xs[j],ys[j]+delta) - rbf(xs[j],ys[j]-delta))/ddelta
            gd = np.array([gx,gy])

            #get and update force
            force = alp*diff*(1/np.linalg.norm(gd))*gd
            sp_forces[j] = sp_forces[j]+force

    # normalize forces by cctr
    for i in range(N):
        if cctr[i] != 0:
            sp_forces[i] = sp_forces[i] / cctr[i]

    # # Plotting code starts
    # xi = np.linspace(min(xs), max(xs), 100)
    # yi = np.linspace(min(ys), max(ys), 100)
    # XI, YI = np.meshgrid(xi, yi)
    # rbf = Rbf(xs, ys, depths, function='thin-plate')
    # ZI = rbf(XI, YI)
    # gy, gx = np.gradient(ZI, .2, .2)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.pcolor(XI, YI, ZI, cmap=cm.jet)
    # plt.scatter(xs, ys, 100, depths, cmap=cm.jet)
    # plt.title('RBF interpolation - multiquadrics')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.colorbar()
    #
    # gs = np.array(sp_forces)
    # plt.subplot(1,2,2)
    # ax = plt.gca()
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(0, 1)
    # ax.quiver(xs, ys,  gs[:,0], gs[:,1])
    # plt.show()
    # # Plotting code ends

    # exit(0)



    return sp_forces

def get_spline_ordered_mds(pos2d, depths, alpha, N_i=1):
    """

    Args:
        pos2d: The MDS positions in 2D.
        depths: Array of depths.
        alpha: Alpha for determining the band sizes.
        N_i: Number of max iterations

    Returns:

    """
    cur_pos = copy.deepcopy(pos2d)
    N = len(depths)
    forces = [np.array([0,0])] * N

    mds_dist_mat = list_to_dist_mat(pos2d)
    sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])
    sorted_depths = depths[sorted_inds]

    for iter in range(N_i):

        # Get the spline forces
        # sp_forces = get_spline_forces_lou(cur_pos,depths)
        sp_forces,vis_vert_rbf = get_spline_forces_pp(cur_pos,depths)
        #sp_forces = get_spline_forces_bp(cur_pos,depths,alpha)


        # Get the stress forces
        cur_dist_mat = list_to_dist_mat(cur_pos)
        dis_forces = get_dist_forces(mds_dist_mat=mds_dist_mat, cur_dist_mat=cur_dist_mat, cur_pos=cur_pos)


        # # Plotting code starts

        xs = [pos[0] for j,pos in enumerate(cur_pos)]
        ys = [pos[1] for j,pos in enumerate(cur_pos)]
        xi = np.linspace(min(xs)-1, max(xs)+1, 100)
        yi = np.linspace(min(ys)-1, max(ys)+1, 100)



        plt.figure()

        # spline gradient forces
        plt.subplot(2, 2, 1)
        plt.axis('equal')
        plt.title('Current interpolation and gradient')
        XI, YI = np.meshgrid(xi, yi)
        rbf = Rbf(xs, ys, depths, function='thin-plate')
        ZI = rbf(XI, YI)
        plt.pcolor(XI, YI, ZI, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
        plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
        plt.colorbar()
        cp = plt.contour(XI, YI, ZI, levels=sorted_depths)
        plt.clabel(cp, inline=True,fontsize=10)
        ax = plt.gca()
        for i2 in range(len(xs)):
            ax.annotate(str(i2), (xs[i2],ys[i2]))

        #get gradient
        gd = []
        delta = 0.01
        for j in range(len(xs)):
            ddelta = 2*delta # double delta
            gx = (rbf(xs[j]+delta,ys[j]) - rbf(xs[j]-delta,ys[j]))/ddelta
            gy = (rbf(xs[j],ys[j]+delta) - rbf(xs[j],ys[j]-delta))/ddelta
            gd.append(np.array([gx,gy]))

        gd = np.array(gd)
        scale = 400
        q = ax.quiver(xs,ys,gd[:,0],gd[:,1],angles='xy',scale=scale,color='black')

        # dist forces
        plt.subplot(2,2,2)
        plt.axis('equal')
        plt.title('Current forces spring(green) and spline(blue)')
        ax = plt.gca()
        dfs = np.array(dis_forces)
        sfs = np.array(sp_forces)
        plt.xlim(xi[0]-1,xi[-1]+1)
        plt.ylim(yi[0]-1,yi[-1]+1)
        plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
        ax.quiver(xs, ys,  dfs[:,0], dfs[:,1],angles='xy',scale=scale,color='g')
        ax.quiver(xs, ys,  sfs[:,0], sfs[:,1],angles='xy',scale=scale,color='b')
        for i2 in range(len(xs)):
            ax.annotate(str(i2), (xs[i2],ys[i2]))

        # Vertex spline
        vis_vert_id = 0
        plt.subplot(2,2,3)
        plt.axis('equal')
        ax = plt.gca()
        plt.title('Spline and force for: ')
        vv_ZI = vis_vert_rbf(XI,YI)
        plt.pcolor(XI, YI, vv_ZI, cmap=cm.jet, vmin=np.amin(vv_ZI), vmax=np.amax(vv_ZI))
        plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(vv_ZI), vmax=np.amax(vv_ZI))

        v_gx = (vis_vert_rbf(xs[vis_vert_id]+delta,ys[vis_vert_id]) - vis_vert_rbf(xs[vis_vert_id]-delta,ys[vis_vert_id]))/ddelta
        v_gy = (vis_vert_rbf(xs[vis_vert_id],ys[vis_vert_id]+delta) - vis_vert_rbf(xs[vis_vert_id],ys[vis_vert_id]-delta))/ddelta
        dprint(v_gx,v_gy)
        ax.quiver(xs[vis_vert_id], ys[vis_vert_id],  [v_gx], [v_gy],angles='xy',scale=scale,color='g')
        plt.colorbar()
        cp = plt.contour(XI, YI, vv_ZI, levels=sorted_depths)
        plt.clabel(cp, inline=True,fontsize=10)


        # following two lines are non vis code also needed for vis
        for vid in range(N):
            forces[vid] = (sp_forces[vid] + dis_forces[vid])/2

        # showing total forces
        plt.subplot(2,2,4)
        plt.axis('equal')
        plt.title('Total forces')
        tfs = np.array(forces)
        ax = plt.gca()
        plt.xlim(xi[0]-1,xi[-1]+1)
        plt.ylim(yi[0]-1,yi[-1]+1)
        plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
        ax.quiver(xs, ys,  tfs[:,0], tfs[:,1],angles='xy',scale=scale,color='g')


        plt.show()
        exit(0)

        plt.subplot(2,3,2)
        plt.title('Current spring(dist) forces')
        ax = plt.gca()
        dfs = np.array(dis_forces)
        ax.quiver(xs, ys,  dfs[:,0], dfs[:,1])

        plt.subplot(2,3,3)
        plt.title('Current spline(gradient) forces')
        ax = plt.gca()
        sfs = np.array(sp_forces)
        # plt.axis('equal')
        ax.quiver(xs, ys,  sfs[:,0], sfs[:,1])

        plt.subplot(2,3,4)
        plt.title('Only colorbar')
        plt.colorbar()



        plt.subplot(2,3,6)
        plt.title('Current spline gradients')
        gy, gx = np.gradient(ZI, .2, .2)
        ax = plt.gca()
        ax.quiver(XI, YI,  gx, gy)

        # following two lines are non vis code also needed for vis
        for vid in range(N):
            forces[vid] = (sp_forces[vid] + dis_forces[vid])/2

        plt.subplot(2,3,5)
        plt.title('Current total forces')
        ax = plt.gca()
        tfs = np.array(forces)
        ax.quiver(xs, ys,  tfs[:,0], tfs[:,1])

        plt.suptitle('Before iteration: '+str(iter))
        plt.show(block=False)



         # Plotting code ends *** A line of nonplotting code in middle to sum forces ***


        ###################################################
        # for each vertex disp is prop to sum of the forces
        ###################################################
        for vid in range(N):
            cur_pos[vid] = cur_pos[vid] + forces[vid]

    # plt.show()
    return cur_pos


def compute_depth_forces_from_spline(cur_pos,res,XI2,YI2,ZIcm,depths,alp):
    """

    Args:
        cur_pos:
        res:
        XI2:
        YI2:
        ZIcm:

    Returns:

    """
    # get point inds on grid
    cur_ids = ipos.get_pos_on_grid2(cur_pos,res,XI2,YI2)

    # compute gradients at the point positions in the cur_ids using central differences
    gd = []
    dpfs = []

    # dprint(depths_curpos)
    for j in range(len(cur_ids)):

        ix = cur_ids[j][1];iy = cur_ids[j][0] # swapped ix and iy to make consistent with np indexing/fetching
        gy = (ZIcm[ix+1,iy] - ZIcm[ix-1,iy])/2
        gx = (ZIcm[ix,iy+1] - ZIcm[ix,iy-1])/2
        gd.append(np.array([gx,gy]))
        # dprint(ZIcm[ix,iy]) # checkpoint1


    # compute the forces based on the diff between the depths, and the gradients
        diff = ZIcm[ix,iy]-depths[j]
        # dpfs.append(alp*diff*(1/(np.linalg.norm(gd[j])))*gd[j])
        dpfs.append(2*diff*gd[j])

    dpfs = np.array(dpfs)
    return dpfs

def get_depth_forces(depths, cur_pos, kernel, res, ZIcmb, XI2b, YI2b, depths_curpos, bw,alp):
    """Compute the depth forces. Builds a monotonic spline based on the cur_pos
    positions to compute the expected depths.

    Args:
        depths: depths on the original space
        cur_pos:

    Returns:
        dep_forces (list): list of forces(np array) for each point.
        monofit (2d array): monotonized current depth interpolated field.

    """

    if ZIcmb is None:
        # compute depths based on cur_pos
        analyser = hsa.analyzer(members=cur_pos, grid_members=None, kernel=kernel) # gridpoints2d is vestigial here
        depths_curpos, all_proj, inside_list_pre = analyser.get_depths() # allproj is vestigial here

        sorted_inds = sorted(range(depths_curpos.shape[0]), key=lambda k: depths_curpos[k])
        median_pos = copy.deepcopy(cur_pos[sorted_inds[-1]])
        #dprint(sorted_inds[-1], depths_curpos)
        #dprint('median', sorted_inds[-1])
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
        phis = np.linspace(-np.pi, np.pi, res)
        rhos = np.linspace(0, max_dist_from_median*1.2, res)
        PI, RI = np.meshgrid(phis, rhos)
        ZI_pol = misc.get_depths_for_polar_grid(PI,RI,tps)

        # monotonize polarized spline
        monofit = monocol.MonotonicityColwise(RI, ZI_pol)
        ZI_pol_mono = monofit(bw=bw)

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
        xs2 = np.linspace(xmin_car,xmax_car,res)
        ys2 = np.linspace(ymin_car,ymax_car,res)
        XI2,YI2 = np.meshgrid(xs2,ys2)


        ZI_pol_mono_flat = ZI_pol_mono.flatten()
        ZI_pol_mono_flat = ZI_pol_mono_flat.reshape(np.shape(X_flat))
        ZI_car_mono = griddata(np.hstack((X_flat,Y_flat)), ZI_pol_mono_flat, (XI2, YI2), method='cubic')
        ZI_car_mono = ZI_car_mono.reshape((res,res))

        ZIcm = np.nan_to_num(ZI_car_mono) # just making the name shorter for convinience.


    else:
        ZIcm = ZIcmb
        XI2 = XI2b
        YI2 = YI2b

    # compute the depth forces from monotonize spline, cur_pos, and original depths
    dpfs = compute_depth_forces_from_spline(cur_pos,res,XI2,YI2,ZIcm,depths,alp)

    return dpfs,ZIcm,XI2,YI2, cur_pos, depths_curpos


def get_depth_engergy(depths, ZIcm, XI2, YI2, cur_pos, res):
    """Get the curre

    Args:
        ZIcm (2d array):
        cur_ids (list):

    Returns:
        depth_energy (float):

    """
    cur_ids = ipos.get_pos_on_grid2(cur_pos,res, XI2, YI2)
    depths_curpos = np.zeros(len(cur_ids))

    for i in range(len(cur_ids)):
        depths_curpos[i] = ZIcm[cur_ids[i][1], cur_ids[i][0]]




    depth_energy = np.sqrt(np.sum((depths - depths_curpos)**2))

    return depth_energy

def get_B_matrix(N, mds_dist_mat, cur_dist_mat):
    """Construct B matrix from http://tosca.cs.technion.ac.il/book/handouts/Stanford09_mds.pdf

    Args:
        mds_dist_mat: Ideal distance matrix.
        cur_dist_mat: Current distance matrix.

    Returns:

    """

    B = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i != j and cur_dist_mat[i,j]>10e-2:
                B[i,j] = -mds_dist_mat[i,j]/cur_dist_mat[i,j]
            elif i!=j:
                B[i,j] = 0

    for i in range(N):
        B[i,i] = -(np.sum(B[i,:]) - B[i,i])

    return B


def get_dist_forces(mds_dist_mat, cur_dist_mat, cur_pos, k=1):
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
    # dist_forces = [np.array([0,0])] * N
    # combs = combinations(range(N),2)
    # energies = [[],[]]
    #
    # for comb in combs:
    #     A = cur_pos[comb[0]]
    #     B = cur_pos[comb[1]]
    #     p = B - A
    #     diff = cur_dist_mat[comb[0],comb[1]] - mds_dist_mat[comb[0],comb[1]]
    #     diffratio = diff/mds_dist_mat[comb[0],comb[1]]
    #     df_A = k*diffratio*p
    #     df_B = -k*diffratio*p
    #
    #     dist_forces[comb[0]] = dist_forces[comb[0]] + df_A
    #     dist_forces[comb[1]] = dist_forces[comb[1]] + df_B

    Z = np.array(cur_pos)
    V = -1*np.ones((N,N))
    np.fill_diagonal(V, N-1)

    B = get_B_matrix(N,mds_dist_mat, cur_dist_mat)
    grad_sig = 2*(np.dot(V,Z)-np.dot(B,Z))

    # dprint(grad_sig)


    # dprint(dist_forces)



    # exit(0)
    # return dist_forces
    return grad_sig

def get_mds_dc(depths, points2d_mds, kernel, res, bw, N_i, max_step, show_fig,alp,spline_lag):
    """

    Args:
        depths: Depths in the original (possibly hd) space.
        points2d_mds: Positions after being projected to 2d.
        alpha: For determining the bands and the outliers.
        N_i: Number of iterations.

    Returns:

    """

    cur_pos = copy.deepcopy(points2d_mds)

    forces = [[],[],[]]

    mds_dist_mat = list_to_dist_mat(points2d_mds)
    # sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])
    # sorted_depths = depths[sorted_inds]

    # compute initial energies
    # first the depth energy
    dep_forces, ZIcm, XI2, YI2, cur_pos, depths_curpos = get_depth_forces(depths, cur_pos, kernel, res, None, None, None, None, bw,alp)
    depth_energy = get_depth_engergy(depths, ZIcm, XI2, YI2, cur_pos, res)
    forces[0].append(depth_energy)

    # next the dist energy and total
    cur_dist_mat = list_to_dist_mat(cur_pos) # need  not be centered for this, but centered version is ok too.
    dis_energy = np.sqrt(np.sum(((mds_dist_mat-cur_dist_mat)/2)**2))
    forces[1].append(0)
    forces[2].append(depth_energy+dis_energy)

    # plotting depth color coded positions before first iteration
    xs = [pt[0] for pt in cur_pos]
    ys = [pt[1] for pt in cur_pos]
    plt.figure()
    plt.axis('equal')
    plt.title('Initial positions')
    plt.scatter(xs, ys, 30, depths, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
    plt.colorbar()
    ax = plt.gca()
    for i2 in range(len(xs)):
        ax.annotate(str(i2), (xs[i2],ys[i2]))
    if show_fig:
            plt.show(block=False)
    else:
        plt.savefig('output/_0.png', bbox_inches='tight')
    # plotting ends


    for iter in range(N_i):

        # compute the depth forces (can lag the depth force to every kth iteration)
        if iter%spline_lag==0:
            dep_forces, ZIcm, XI2, YI2, cur_pos, depths_curpos = get_depth_forces(depths, cur_pos, kernel, res, None, None, None,depths_curpos,bw,alp)
        else:
            dep_forces = compute_depth_forces_from_spline(cur_pos,res,XI2,YI2,ZIcm,depths,alp=alp)
        dpfs = np.array(dep_forces)
        xs = [pos[0] for j,pos in enumerate(cur_pos)] # after centering the cur_pos about median
        ys = [pos[1] for j,pos in enumerate(cur_pos)]


        # compute the distance forces and add to forces
        cur_dist_mat = list_to_dist_mat(cur_pos)
        dis_forces = get_dist_forces(mds_dist_mat=mds_dist_mat, cur_dist_mat=cur_dist_mat, cur_pos=cur_pos, k=1)
        dfs = np.array(dis_forces)



        # make the move
        # tfs = -dpfs
        tfs = -(100*dpfs + dfs)

        tfs = 0.1*tfs

        # norms = np.linalg.norm(tfs, axis=1)
        # pos = np.where(norms>max_step)
        # tfs[pos] = (tfs[pos]/norms[pos,None])*max_step
        # for j in range(len(cur_pos)):
        #     cur_pos[j] = cur_pos[j] + tfs[j,:]

        norms = np.linalg.norm(tfs, axis=1)
        pos = np.where(norms>max_step)
        tfs[pos] = (tfs[pos]/norms[pos,None])*max_step
        for j in range(len(cur_pos)):
            cur_pos[j] = cur_pos[j] + tfs[j,:]


        # # compute energies after position update
        depth_energy = get_depth_engergy(depths, ZIcm, XI2, YI2,  cur_pos, res)
        dis_energy = np.sqrt(np.sum(((mds_dist_mat-cur_dist_mat)/2)**2))

        forces[0].append(depth_energy)
        forces[1].append(dis_energy)
        forces[2].append(depth_energy+dis_energy)

        dprint(depths_curpos, depths)

        # vis


        xs_new = [pos[0] for j,pos in enumerate(cur_pos)]
        ys_new = [pos[1] for j,pos in enumerate(cur_pos)]

        arrow_scale = 1

        plt.figure()
        plt.suptitle('Iteration: '+str(iter))

        plt.subplot(2,2,1)
        plt.axis('equal')
        plt.title('Monotonized TPS')
        plt.pcolor(XI2, YI2, ZIcm, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
        plt.scatter(xs, ys, 30, depths_curpos, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
        plt.colorbar()
        CS = plt.contour(XI2, YI2, ZIcm, levels=[1,2,3], colors='k')
        plt.clabel(CS, inline=1, fontsize=10)
        ax = plt.gca()
        for i2 in range(len(xs)):
            ax.annotate(str(i2), (xs[i2],ys[i2]))


        plt.subplot(2,2,2)
        plt.axis('equal')
        plt.title('Forces: dis(red), dep(green)')
        plt.scatter(xs, ys, 30, depths, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
        plt.colorbar()
        ax = plt.gca()
        ax.quiver(xs, ys,  -dpfs[:,0], -dpfs[:,1],angles='xy',scale=arrow_scale,color='g')
        ax.quiver(xs, ys,  -dfs[:,0], -dfs[:,1],angles='xy',scale=arrow_scale,color='r')
        ax.quiver(xs, ys,  tfs[:,0], tfs[:,1],angles='xy',scale=arrow_scale,color='black',headwidth=6)
        for i2 in range(len(xs)):
            ax.annotate(str(i2), (xs[i2],ys[i2]))

        plt.subplot(2,2,3)
        plt.axis('equal')
        plt.title('New positions')
        plt.scatter(xs_new, ys_new, 30, depths, cmap=cm.jet, vmin=np.amin(ZIcm), vmax=np.amax(ZIcm))
        plt.colorbar()
        ax = plt.gca()
        for i2 in range(len(xs)):
            ax.annotate(str(i2), (xs_new[i2],ys_new[i2]))

        plt.subplot(2,2,4)
        # plt.axis('equal')
        plt.title('Energies - dist, depth, total')
        plt.plot(range(iter+2),forces[2],'black',label="total")
        plt.plot(range(iter+2),forces[0],'g--',label="depth")
        plt.plot(range(iter+2),forces[1],'r--',label="dist")
        plt.legend(loc=2)
        ax = plt.gca()
        ax.set_ylim([-1,max(forces[2])+1])

        # showFig = False
        if show_fig:
            plt.show(block=False)
        else:
            plt.savefig('output/'+str(iter)+'.png', bbox_inches='tight')

    if show_fig:
        plt.show()

    return cur_pos



