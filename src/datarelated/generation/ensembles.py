"""Created on May 5, 2016

Functions to generate various kinds of ensembles.

"""
import copy
import matplotlib.pyplot as plt
import numpy as np
from infpy.gp import GaussianProcess, gp_1D_X_range

from libs.productivity import dprint
import src.datarelated.generation.infgpext as gpe
import seaborn as sns
import math
from scipy import linalg
from scipy.spatial.distance import mahalanobis as mahalanobis
from scipy.spatial.distance import euclidean
from scipy.stats import logistic

def centeroidnp(arr):
    """

    Args:
        arr:

    Returns:
    References:
        http://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_point2d_ensemble(N):
    """
    Args:
        N: Ensemble size

    Returns:
        points: A list of points

    """
    points_list  = []
    rand_points = np.random.rand(N,2)
    rand_points = np.around(rand_points, 3)

    # cen = centeroidnp(rand_points)
    # dprint(rand_points)
    # rand_points[:,0] = rand_points[:,0] - cen[0]
    # rand_points[:,1] = rand_points[:,1] - cen[1]
    # dprint(cen)
    # dprint(rand_points)


    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list




def get_points2d_ensemble_normal(N):
    """Gets points from a normal distribution.

    Args:
        N: Ensemble size

    Returns:
        points: A list of points

    """

    points_list = []

    u = [0.5,0.5]
    covar = [[1,0],[0,3]]
    rand_points = np.random.multivariate_normal(u, covar, N)
    rand_points = np.around(rand_points, 3)

    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)

    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list


def get_points2d_ensemble_normal_hd(N, u, covar):
    """Gets points from anisotropic normal distribution.
    date: 19 feb 2017
    Returns:

    """
    dprint(covar)
    points_list = []


    rand_points = np.random.multivariate_normal(u, covar, N)
    rand_points = np.around(rand_points, 3)

    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)

    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list

def get_uniform_ensemble(us, covs, N):
    """

    Args:
        u:
        covs:
        N:

    Returns:

    """
    samples = []
    d = len(us)
    for i in range(d):
        hf = (covs[i][i])/float(2)
        u = us[i]

        s = np.random.uniform(u-hf, u+hf, N)

        samples.append(s)

    samples = np.array(samples)

    dprint(np.shape(samples))
    points = []
    for j in range(N):
        points.append(samples[:,j])

    return points






def get_multimodal_normal_hd(Ns, us, covs):
    """Gets points a multimodal distribution.

    Args:
        Ns: List of N for each mode.
        us: list of means.
        covs: list of covariance matrices.

    Returns:

    """
    points_list = []


    N = Ns[0]
    u = us[0]
    cov = covs[0]

    rand_points = np.random.multivariate_normal(u, cov, N)
    rand_points = np.around(rand_points, 3)

    # rand_points = rand_points - np.amin(rand_points)
    # rand_points = rand_points / np.amax(rand_points)

    for i in range(1,len(Ns)):
        N = Ns[i]
        u = us[i]
        cov = covs[i]

        rand_points2 = np.random.multivariate_normal(u, cov, N)
        rand_points2 = np.around(rand_points2, 3)

        rand_points = np.concatenate((rand_points, rand_points2),)

    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)

    for j in range(sum(Ns)):
        points_list.append(rand_points[j,:])

    return points_list


def get_rings_sin(N, us, covs, freq):
    """

    Args:
        Ns:
        us:
        covs:
        lamb: factor controlling the wavelength of rings

    Returns:

    """

    init_fac = 5 # a buffer factor to make sure enough points left after sampleing

    X = get_points2d_ensemble_normal_hd(init_fac*N,us,covs)
    VI = linalg.inv(np.cov(np.array(X).T))

    rands = np.random.rand(init_fac*N)


    u = np.array(us)
    accepted_list = []


    for i in range(init_fac*N):
        mds = mahalanobis(u, X[i], VI)
        sinval = np.fabs(np.cos(freq*mds))
        if rands[i]<sinval:
            accepted_list.append(X[i])
            if len(accepted_list)==N:
                break

    return accepted_list


def get_rings_cos(N, us, covs, freq):
    """

    Args:
        Ns:
        us:
        covs:
        lamb: factor controlling the wavelength of rings

    Returns:

    """

    init_fac = 5 # a buffer factor to make sure enough points left after sampleing

    X = get_points2d_ensemble_normal_hd(init_fac*N,us,covs)
    VI = linalg.inv(np.cov(np.array(X).T))

    rands = np.random.rand(init_fac*N)


    u = np.array(us)
    accepted_list = []

    for i in range(init_fac*N):
        mds = mahalanobis(u, X[i], VI)
        cosval = np.fabs(np.cos(freq*mds))
        if rands[i]<cosval:
            accepted_list.append(X[i])
            if len(accepted_list)==N:
                break

    return accepted_list

def get_holed_dist(N, us, covs, holeloc, sizefac):
    """

    Args:
        N:
        us:
        covs:
        holeloc: location of hole
        sizefac: controls size of hole

    Returns:

    """

    X = get_points2d_ensemble_normal_hd(N,us,covs)

    accepted_list = []
    rands = np.random.rand(N)
    dists = np.zeros(N)
    for i in range(N):
        dists[i] = euclidean(holeloc,X[i])

    dists = dists/max(dists)
    for i in range(N):
        # if rands[i] < sizefac*dists[i]:
        if 0.5 < sizefac*dists[i]:

            accepted_list.append(X[i])

        else:
            dprint(i)



    return accepted_list

def get_seperating_hole_dist(N,u, cov, axid, axloc, dist):
    """Generates a normal distribution and a separating hole.

    Args:
        N:
        u:
        cov:
        axisid:
        dist:

    Returns:

    """

    X = get_points2d_ensemble_normal_hd(N,u,cov)
    Y = np.array(X)
    dprint(np.ptp(Y[:,0]), np.ptp(Y[:,1]))
    dprint(np.min(X,axis=0), np.amax(X,axis=0))
    points = []

    for i in range(N):

        if abs(X[i][axid] - axloc)>dist:
            points.append(X[i])

    Y = np.array(points)
    dprint(np.ptp(Y[:,0]), np.ptp(Y[:,1]))
    dprint(np.min(X,axis=0), np.amax(X,axis=0))

    dprint(len(points))

    return points

def get_cube_uniform(N, d, c):
    """

    Args:
        N: number of points
        d: dimensions
        c: center translation

    Returns:
        points_list:
    """

    s = np.random.uniform(-0.5, 0.5, (N,d))
    points_list = []

    for i in range(N):
        points_list.append(s[i,:d])

    return points_list

def get_threequartercube_uniform(N, d, c):
    """Get an ell shape of three quarters cube.

    Args:
        N: number of points
        d: dimensions
        c: center

    Returns:
        points_list
    """

    s = np.random.uniform(-0.5, 0.5, (N,d))
    points_list = []

    for i in range(N):
        if sum(s[i,:d]>0) != d:
            points_list.append(s[i,:d])




    return points_list


def get_center_spaceship(N):
    """Two normal distributions in d-1 space, cris-crossing at the center.

    Returns:

    """
    points_list = []

    u = [0.5,0.5]
    covar = [[1,0],[0,10]]
    rand_points1 = np.random.multivariate_normal(u, covar, N/2)


    u = [0.5,0.5]
    covar = [[10,0],[0,1]]
    rand_points2 = np.random.multivariate_normal(u, covar, N/2)

    rand_points = np.vstack([rand_points1,rand_points2])


    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)

    rand_points = np.around(rand_points, 3)

    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list

def get_side_spaceship():
    """Three normal distributions in d-1 space, two cris-crossing the third.

    Returns:

    """
    pass

def get_points2d_ensemble_normal_multimodal(N):
    """Gets points from a two 2d normal distribution.

    Args:
        N: Ensemble size

    Returns:
        points: A list of points

    """

    points_list = []

    u1 = [0.5,0.5]
    u2 = [0.5,2.0]
    covar = [[0.5,0],[0,0.5]]
    rand_points = np.random.multivariate_normal(u1, covar, N/2)
    rand_points = np.around(rand_points, 3)

    rand_points2 = np.random.multivariate_normal(u2, covar, N-N/2)
    rand_points2 = np.around(rand_points2, 3)

    rand_points = np.concatenate((rand_points, rand_points2),)

    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)


    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list

def get_points3d_ensemble_normal(N):
    """Gets points from a normal distribution.

    Args:
        N: Ensemble size

    Returns:
        points: A list of points

    """

    points_list = []

    u = [0,0,0]
    covar = [[1,0,0],[0,2,0],[0,0,5]]
    rand_points = np.random.multivariate_normal(u, covar, N)
    rand_points = np.around(rand_points, 3)

    rand_points = rand_points - np.amin(rand_points)
    rand_points = rand_points / np.amax(rand_points)

    for i in range(N):
        points_list.append(rand_points[i,:])

    return points_list

def get_model5_shape_contamination_ensemble(dims, N1,N2,mu1,mu2):
    """Generates a shape contaminated ensemble based on model 5 in the paper
    'On the concept of depth for functional data-Lopez-Pintado et al.

    Args:
        dims (int): number of dimensions/support length
        N1 (int): number of type 1 functions
        N2 (int): number of type 2 functions
        mu1 (float): mu parameter for type 1 functions
        mu2 (float): mu parameter for type 2 functions

    Returns:
        functions: A list of functions

    """

    def fn(x):
        return 4*x

    functions = []
    interval = 1.0/dims
    support = gp_1D_X_range(0, 1, interval)

    k1 = gpe.CustomSquaredExponentialKernel([1],None,None,1,1,mu1)
    gp1 = GaussianProcess([], [], k1)

    functions = gpe.gp_get_samples_from(gp1,support,fn, N1)


    k2 = gpe.CustomSquaredExponentialKernel([1],None,None,1,1,mu2)
    gp2 = GaussianProcess([], [], k2)
    functions.extend(gpe.gp_get_samples_from(gp2,support,fn, N2))

    return functions

def get_shifted_sinusoids(mu, C, a, N, dims):
    """Shifted sinusoids based on functions in Mahsa's paper

    Args:
        mu: Mean
        C: covariance matrix
        a: amplitude
        N: ensemble size
        dims: number of dimensions

    Returns:

        functions: A list of functions

    """

    functions = []




    interval = 120/dims
    support = gp_1D_X_range(0, 120, interval)
    support = support * 0.0174
    support_sq = np.squeeze(support)
    rand_pts = np.random.multivariate_normal(mu, C, N)
    # rand_points = np.around(rand_points, 3)

    for i in range(N):
        base = a*np.sin(support_sq+rand_pts[i,0])
        base = base + rand_pts[i,1]
        functions.append(base)
    return functions,rand_pts


def get_function_ensemble(N,dims):
    """
    Args:
        N: Ensemble size

    Returns:
        functions: A list of functions
    """

    functions = []

    cs = np.random.normal(size=N)

    #dims = 120
    interval = 20.01/dims

    support = gp_1D_X_range(-10, 10.01, interval)
    dprint(len(support))

    support_sq = np.squeeze(support)
    for i in range(N):
        y = 0.5*support_sq + cs[i]

        for j in range(len(support_sq)):
            y[j] = y[j]+np.random.rand()

        functions.append(y)

    return functions

def get_function_ensemble_collinear(N,dims):
    """These functions are co-linear in high dimensional space.

    Args:
        N: Ensemble size

    Returns:
        functions: A list of functions
    """

    functions = []

    cs = np.random.normal(size=N)

    #dims = 120
    interval = 20.01/dims

    support = gp_1D_X_range(-10, 10.01, interval)
    dprint(len(support))

    support_sq = np.squeeze(support)
    for i in range(N):
        y = support_sq * cs[i]

        for j in range(len(support_sq)):
            y[j] = y[j]+np.random.rand()

        functions.append(y)
        dprint(y)
    dprint(functions)
    return functions

def get_point2d_gram_ensemlbe(points2d, kernel):
    """Generates a 2D point ensemble and returns the gram matrix using kernel

    Args:
        points2d: list of ensemble members

    Returns:
        K: Gram matrix
        K_proj: Euclidean 2d projections of the points in the gram matrix

    """
    N = len(points2d)
    K = np.zeros(shape=(N,N))

    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(points2d[i],points2d[j])
            #K[j,i] = kernel(points2d[j],points2d[i])

    K_proj = points2d

    return K, K_proj

def get_point2d_test_gram_ensemble(pts, gridpts, kernel):
    """Generates the 'gramlike' matrices with the dot products between the
    points and the test points which are grid

    Args:
        pts: The points forming the ellipses.
        gridpts: The grid points to be checked for falling inside the ellipses.
        kernel: The kernel that computes the dot product.

    Returns:
        Ktest: matrix with dot products of gridpts and pts
        Kxx:  matrix with dot products of grid pts
        Kxx_proj: Euclidean 2d projections of the points in the gridpts.

    """

    Kxx_proj = copy.deepcopy(pts)
    Kxx_proj.extend(gridpts)
    N_Kxx = len(Kxx_proj)
    Kxx = np.zeros(shape=(N_Kxx,N_Kxx))

    for i in range(N_Kxx):
        for j in range(N_Kxx):
            Kxx[i,j] = kernel(Kxx_proj[i],Kxx_proj[j])
            # Kxx[j,i] = kernel(Kxx_proj[j],Kxx_proj[i])

    N = len(pts)

    Ktest = np.zeros(shape=(N,N_Kxx))
    for i in range(N):
        for j in range(N_Kxx):
                Ktest[i,j] = kernel(pts[i],Kxx_proj[j])
    dprint(np.shape(Kxx))

    return Ktest,Kxx,Kxx_proj

def get_points_on_grid(res):
    """

    Args:
        res: Resolution of grid points.

    Returns:
        gridpoints2d: A list of grid points
    """

    gridpoints2d = []

    x_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
    x_poses = np.around(x_poses,3)
    y_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
    y_poses = np.around(y_poses,3)

    for x in x_poses:
        for y in y_poses:
            gridpoints2d.append(np.array([x,y]))

    return gridpoints2d


def get_multimodel_normal_points(N, u1,C1,u2,C2):
    """Generates and returns multimodal normal points in the 2D array format. This version
    is for the 2016-09-13 monotonicity experiments. Dimension wise normalization added.
    And center adjustment.

    Args:
        N:

    Returns:

    """

    center = np.array((u1+u2)/2)

    rand_points = np.random.multivariate_normal(u1, C1, N/2)
    rand_points = np.around(rand_points, 3)

    rand_points2 = np.random.multivariate_normal(u2, C2, N-N/2)
    rand_points2 = np.around(rand_points2, 3)

    rand_points = np.concatenate((rand_points, rand_points2),)


    minvals = np.amin(rand_points,axis=0)
    rand_points = rand_points - minvals
    center = center - minvals

    maxvals = np.amax(rand_points,axis=0)
    rand_points = rand_points / maxvals
    center = center / maxvals

    dprint(center)
    return rand_points,center


def get_multimodal_exp_points(N):
    """Gets points from a 2D exp distribution

    Returns:

    """

    center = np.array([0,0])

    rand_points = np.random.exponential(size=(N/2,2))
    rand_points = np.around(rand_points, 3)

    rand_points2 = -np.random.exponential(size=(N-N/2,2))
    rand_points2 = np.around(rand_points2, 3)

    rand_points = np.concatenate((rand_points, rand_points2),)

    minvals = np.amin(rand_points,axis=0)
    rand_points = rand_points - minvals
    center = center - minvals
    maxvals = np.amax(rand_points,axis=0)
    rand_points = rand_points / maxvals
    center = center/maxvals


    return rand_points,center

def get_multimodal_cosexp_points(N):
    """ Get points from  a cos*exp dist using rejection sampling.

    Args:
        N:

    Returns:

    """
    center = np.array([0,0])

    x = np.linspace(0, 1, 100)

    xs = np.random.rand(20000)
    ys = np.random.rand(20000)

    cosexp = xs*(1-np.cos(10*xs))*np.exp(xs)

    cosexp2 = x*(1-np.cos(10*x))*np.exp(x)


    # plt.plot(cosexp2)
    # plt.show()



    samples = []
    max = 1*(1-np.cos(1))*np.exp(1)
    dprint(max,'max')

    for i in range(len(xs)):
        if ys[i]<=cosexp[i]/max:
            samples.append(xs[i])

        if len(samples) == 2*N:
            break

    # sns.distplot(samples)
    # plt.show()
    # dprint(len(samples),'sample length', i)



    rand_points = np.zeros((N,2))
    rand_points[:,0] = samples[:N]
    rand_points[:,1] = samples[N:2*N]

    rand_points[:N/2] *= -1

    minvals = np.amin(rand_points,axis=0)
    rand_points = rand_points - minvals
    center = center - minvals
    maxvals = np.amax(rand_points,axis=0)
    rand_points = rand_points / maxvals
    center = center/maxvals

    return rand_points,center



def get_points_on_grid_2Darray(res):
    """This version is for the 2016-09-13 monotonicity experiments. Data is
    normalized and return format is different.

    Args:
        res: Resolution of grid points.

    Returns:
        gridpoints2d: A 2D numpy array.
    """

    gridpoints2d = []

    x_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
    x_poses = np.around(x_poses,3)
    y_poses = np.linspace(0.0, 1.0, num=res, endpoint=False)
    y_poses = np.around(y_poses,3)

    xv, yv = np.meshgrid(x_poses, y_poses)

    return xv,yv