"""
Multi-dimensional Scaling (MDS)
"""

# author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# Licence: BSD

import numpy as np
from produtils import dprint
import warnings

from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
# from sklearn.utils import check_random_state, check_arrays
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.isotonic import IsotonicRegression
from scipy.spatial.distance import euclidean
from datarelated.processing.depths import get_median_and_bands
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import src.utils.mathops as mops
import src.utils.vectors as vecs

def project_directed_constraint(X, v, p_id, band_vert_ids, N_i=10):
    """Projecting directed constraint to get the misplaced point out of the band.

    Args:
        X: Input array of point positions
        v: Direction of the constraint, also includes the distance as magnitude
        p_id: The id of the point to be moved out of the band
        band_vert_ids: The ids of the vertices in the band
        N_i: Number of constraint satisfaction iterations

    Returns:

    """

    wp = 1
    wq = len(band_vert_ids)
    d = vecs.length(v)

    w_p = (wq/(wp+wq))
    w_q = (wp/(wp+wq))
    p = X[p_id,:]

    for i in range(N_i):
        for band_id in band_vert_ids:
            q = X[band_id]
            pq = vecs.vector(p,q)
            pq_mag = vecs.length(pq)



        # X[p_id,:] = X[p_id,:] + r_p
        #
        # for band_id in band_vert_ids:
        #     X[band_id,:] = X[band_id,:] - r_p

    exit(0)



    return X

def project_constraints(X, N_i=10):
    """

    Args:
        X: Input array of point positions
        N_i: Number of constraint satisfaction iterations

    Returns:
        X_new: New positions after N_i cycles of constraint projection
    """

    X_new = X

    # dprint(euclidean(X[3,:],X[14,:]), euclidean(X[3,:],X[6,:]), euclidean(X[3,:],X[15,:]))

    d = 0.1
    for i in range(N_i):


        p = X_new[3,:]
        q = X_new[14,:]
        pq = q-p
        pq_mag = np.linalg.norm(pq)

        if abs(pq_mag - d)>0.01:
            r = (d - pq_mag)/(2*pq_mag)
            r = r*pq
            X_new[3,:] = p - r
            X_new[14,:] = q + r

    return X_new

def project_constraints_order(X, bands_dict, N_i=100):
    """Projects constraints that the order of the distance from median
    should be proportional to the depth.

    Args:
        X:
        N_i:

    Returns:

    """

    sorted_inds = bands_dict['sorted_inds']
    depths = bands_dict['depths']
    dprint(sorted_inds)
    dprint(depths[sorted_inds])
    ensize = len(sorted_inds)

    d_i_old = depths[sorted_inds[0]]

    X_new = X

    for iter in range(N_i):

        for i in range(1,ensize):
            d_i = depths[sorted_inds[i]]
            # dprint(i, sorted_inds[i], d_i == d_i_old)
            p = X_new[sorted_inds[0],:]
            q = X_new[sorted_inds[i],:]

            pq = q-p
            pq_mag = np.linalg.norm(pq)

            if d_i == d_i_old: # Testing if the depth of member(i) is same as member(i-1)
                # enforce equality constraint - dist equal to prev dist

                if abs(pq_mag - d_i_old)>0.01:
                    rmag = (d_i - pq_mag)/(2*pq_mag)
                    r = rmag*pq
                    X_new[sorted_inds[0],:] = p - r
                    X_new[sorted_inds[i],:] = q + r
            else:
                # enforce inequality constraint - dist more than prev dist
                if pq_mag - d_i_old < 0:
                    r = (d_i - pq_mag)/(2*pq_mag)
                    r = r*pq
                    X_new[sorted_inds[0],:] = p - r
                    X_new[sorted_inds[i],:] = q + r

            d_i_old = d_i



    return X_new

def project_constraints_band(X, bands_dict, N_i=10):
    """Projects constraints that the band members stay inside the respective
    bands.

    Args:
        X:
        N_i:

    Returns:

    """

    # First for band50

    # Get the band points
    band50_inds = bands_dict['band50']
    band100_inds = bands_dict['band100']
    band50pts = X[list(band50_inds)]

    hull = MultiPoint(band50pts).convex_hull

    # hull_band50 = ConvexHull(band50pts)
    # hull_path50 = Path(band50pts[hull_band50.vertices])
    # dprint(hull_band50.vertices)

    # Find points supposed to be in band but not and iterate
    only_band100_inds = band100_inds-band50_inds
    for ind in only_band100_inds:
        # dprint(ind,hull_path50.contains_point(X[ind,:]))
        point = Point(X[ind])
        if hull.contains(point):
            xs,ys = hull.boundary.xy # see shapely.geometry.linestring module
            set2 = {'xs':list(xs), 'ys': list(ys)}

            # find minkowski diff set
            mink_diff = mops.get_minkowski_diff(X[ind], set2)

            # find mpv
            mpv = mops.get_min_penetration_vector(X[ind], mink_diff)

            # do N_i iterations of projection
            X = project_directed_constraint(X,mpv,ind, band50_inds)



    # Repeat for band100



    X_new = X



    # algo for this function.

    return X_new

# def _smacof_single(depths, bands_dict, similarities, metric=True, n_components=2, init=None,
#                    max_iter=300, verbose=0, eps=1e-3, random_state=None):
#     """
#     Computes multidimensional scaling using SMACOF algorithm
#
#     Parameters
#     ----------
#     similarities: symmetric ndarray, shape [n * n]
#         similarities between the points
#
#     metric: boolean, optional, default: True
#         compute metric or nonmetric SMACOF algorithm
#
#     n_components: int, optional, default: 2
#         number of dimension in which to immerse the similarities
#         overwritten if initial array is provided.
#
#     init: {None or ndarray}, optional
#         if None, randomly chooses the initial configuration
#         if ndarray, initialize the SMACOF algorithm with this array
#
#     max_iter: int, optional, default: 300
#         Maximum number of iterations of the SMACOF algorithm for a single run
#
#     verbose: int, optional, default: 0
#         level of verbosity
#
#     eps: float, optional, default: 1e-6
#         relative tolerance w.r.t stress to declare converge
#
#     random_state: integer or n2umpy.RandomState, optional
#         The generator used to initialize the centers. If an integer is
#         given, it fixes the seed. Defaults to the global numpy random
#         number generator.
#
#     Returns
#     -------
#     X: ndarray (n_samples, n_components), float
#                coordinates of the n_samples points in a n_components-space
#
#     stress_: float
#         The final value of the stress (sum of squared distance of the
#         disparities and the distances for all constrained points)
#
#     """
#     n_samples = similarities.shape[0]
#     random_state = check_random_state(random_state)
#
#     if similarities.shape[0] != similarities.shape[1]:
#         raise ValueError("similarities must be a square array (shape=%d)" %
#                          n_samples)
#     res = 100 * np.finfo(np.float).resolution
#     if np.any((similarities - similarities.T) > res):
#         raise ValueError("similarities must be symmetric")
#
#     sim_flat = ((1 - np.tri(n_samples)) * similarities).ravel()
#     sim_flat_w = sim_flat[sim_flat != 0]
#     if init is None:
#         # Randomly choose initial configuration
#         X = random_state.rand(n_samples * n_components)
#         X = X.reshape((n_samples, n_components))
#     else:
#         # overrides the parameter p
#         n_components = init.shape[1]
#         if n_samples != init.shape[0]:
#             raise ValueError("init matrix should be of shape (%d, %d)" %
#                              (n_samples, n_components))
#         X = init
#
#     old_stress = None
#     ir = IsotonicRegression()
#     for it in range(max_iter):
#         # Compute distance and monotonic regression
#         dis = euclidean_distances(X)
#
#         if metric:
#             disparities = similarities
#         else:
#             dis_flat = dis.ravel()
#             # similarities with 0 are considered as missing values
#             dis_flat_w = dis_flat[sim_flat != 0]
#
#             # Compute the disparities using a monotonic regression
#             disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
#             disparities = dis_flat.copy()
#             disparities[sim_flat != 0] = disparities_flat
#             disparities = disparities.reshape((n_samples, n_samples))
#             disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
#                                    (disparities ** 2).sum())
#
#         # Compute stress
#         stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
#
#         # Update X using the Guttman transform
#         dis[dis == 0] = 1e-5
#         ratio = disparities / dis
#         B = - ratio
#         B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
#         X = 1. / n_samples * np.dot(B, X)
#
#         dis = np.sqrt((X ** 2).sum(axis=1)).sum()
#
#         # dprint(it, stress, np.shape(X))
#         # X = project_constraints(X)
#         # X = project_constraints_band(X, bands_dict)
#         X = project_constraints_order(X,bands_dict)
#
#         if verbose == 2:
#             print('it: %d, stress %s' % (it, stress))
#         if old_stress is not None:
#             if(old_stress - stress / dis) < eps:
#                 if verbose:
#                     print('breaking at iteration %d with stress %s' % (it,
#                                                                        stress))
#                 break
#         old_stress = stress / dis
#
#     return X, stress
#
#
# def smacof(depths, alpha, similarities, metric=True, n_components=2, init=None, n_init=8,
#            n_jobs=1, max_iter=300, verbose=0, eps=1e-3, random_state=None):
#     """
#     Computes multidimensional scaling using SMACOF (Scaling by Majorizing a
#     Complicated Function) algorithm
#
#     The SMACOF algorithm is a multidimensional scaling algorithm: it minimizes
#     a objective function, the *stress*, using a majorization technique. The
#     Stress Majorization, also known as the Guttman Transform, guarantees a
#     monotone convergence of Stress, and is more powerful than traditional
#     techniques such as gradient descent.
#
#     The SMACOF algorithm for metric MDS can summarized by the following steps:
#
#     1. Set an initial start configuration, randomly or not.
#     2. Compute the stress
#     3. Compute the Guttman Transform
#     4. Iterate 2 and 3 until convergence.
#
#     The nonmetric algorithm adds a monotonic regression steps before computing
#     the stress.
#
#     Parameters
#     ----------
#     similarities : symmetric ndarray, shape (n_samples, n_samples)
#         similarities between the points
#
#     metric : boolean, optional, default: True
#         compute metric or nonmetric SMACOF algorithm
#
#     n_components : int, optional, default: 2
#         number of dimension in which to immerse the similarities
#         overridden if initial array is provided.
#
#     init : {None or ndarray of shape (n_samples, n_components)}, optional
#         if None, randomly chooses the initial configuration
#         if ndarray, initialize the SMACOF algorithm with this array
#
#     n_init : int, optional, default: 8
#         Number of time the smacof algorithm will be run with different
#         initialisation. The final results will be the best output of the
#         n_init consecutive runs in terms of stress.
#
#     n_jobs : int, optional, default: 1
#
#         The number of jobs to use for the computation. This works by breaking
#         down the pairwise matrix into n_jobs even slices and computing them in
#         parallel.
#
#         If -1 all CPUs are used. If 1 is given, no parallel computing code is
#         used at all, which is useful for debugging. For n_jobs below -1,
#         (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
#         are used.
#
#     max_iter : int, optional, default: 300
#         Maximum number of iterations of the SMACOF algorithm for a single run
#
#     verbose : int, optional, default: 0
#         level of verbosity
#
#     eps : float, optional, default: 1e-6
#         relative tolerance w.r.t stress to declare converge
#
#     random_state : integer or numpy.RandomState, optional
#         The generator used to initialize the centers. If an integer is
#         given, it fixes the seed. Defaults to the global numpy random
#         number generator.
#
#     Returns
#     -------
#     X : ndarray (n_samples,n_components)
#         Coordinates of the n_samples points in a n_components-space
#
#     stress : float
#         The final value of the stress (sum of squared distance of the
#         disparities and the distances for all constrained points)
#
#     Notes
#     -----
#     "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
#     Groenen P. Springer Series in Statistics (1997)
#
#     "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
#     Psychometrika, 29 (1964)
#
#     "Multidimensional scaling by optimizing goodness of fit to a nonmetric
#     hypothesis" Kruskal, J. Psychometrika, 29, (1964)
#     """
#
#     similarities, = check_arrays(similarities, sparse_format='dense')
#     random_state = check_random_state(random_state)
#
#     if hasattr(init, '__array__'):
#         init = np.asarray(init).copy()
#         if not n_init == 1:
#             warnings.warn(
#                 'Explicit initial positions passed: '
#                 'performing only one init of the MDS instead of %d'
#                 % n_init)
#             n_init = 1
#
#     best_pos, best_stress = None, None
#
#     median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths, alpha=alpha)
#     sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])
#
#     bands_dict = {'median':median,
#                   'band50':band50,
#                   'band100':band100,
#                   'outliers':outliers,
#                   'depths': depths,
#                   'sorted_inds': sorted_inds}
#
#
#     if n_jobs == 1:
#         for it in range(n_init):
#             pos, stress = _smacof_single(depths, bands_dict, similarities, metric=metric,
#                                          n_components=n_components, init=init,
#                                          max_iter=max_iter, verbose=verbose,
#                                          eps=eps, random_state=random_state)
#             if best_stress is None or stress < best_stress:
#                 best_stress = stress
#                 best_pos = pos.copy()
#     else:
#         seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
#         results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
#             delayed(_smacof_single)(
#                 similarities, metric=metric, n_components=n_components,
#                 init=init, max_iter=max_iter, verbose=verbose, eps=eps,
#                 random_state=seed)
#             for seed in seeds)
#         positions, stress = zip(*results)
#         best = np.argmin(stress)
#         best_stress = stress[best]
#         best_pos = positions[best]
#     return best_pos, best_stress


# class CMDS(BaseEstimator):
#     """Constrained Multidimensional scaling
#
#     Parameters
#     ----------
#     metric : boolean, optional, default: True
#         compute metric or nonmetric SMACOF (Scaling by Majorizing a
#         Complicated Function) algorithm
#
#     n_components : int, optional, default: 2
#         number of dimension in which to immerse the similarities
#         overridden if initial array is provided.
#
#     n_init : int, optional, default: 4
#         Number of time the smacof algorithm will be run with different
#         initialisation. The final results will be the best output of the
#         n_init consecutive runs in terms of stress.
#
#     max_iter : int, optional, default: 300
#         Maximum number of iterations of the SMACOF algorithm for a single run
#
#     verbose : int, optional, default: 0
#         level of verbosity
#
#     eps : float, optional, default: 1e-6
#         relative tolerance w.r.t stress to declare converge
#
#     n_jobs : int, optional, default: 1
#         The number of jobs to use for the computation. This works by breaking
#         down the pairwise matrix into n_jobs even slices and computing them in
#         parallel.
#
#         If -1 all CPUs are used. If 1 is given, no parallel computing code is
#         used at all, which is useful for debugging. For n_jobs below -1,
#         (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
#         are used.
#
#     random_state : integer or numpy.RandomState, optional
#         The generator used to initialize the centers. If an integer is
#         given, it fixes the seed. Defaults to the global numpy random
#         number generator.
#
#     dissimilarity : string
#         Which dissimilarity measure to use.
#         Supported are 'euclidean' and 'precomputed'.
#
#
#     Attributes
#     ----------
#     ``embedding_`` : array-like, shape [n_components, n_samples]
#         Stores the position of the dataset in the embedding space
#
#     ``stress_`` : float
#         The final value of the stress (sum of squared distance of the
#         disparities and the distances for all constrained points)
#
#
#     References
#     ----------
#     "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
#     Groenen P. Springer Series in Statistics (1997)
#
#     "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
#     Psychometrika, 29 (1964)
#
#     "Multidimensional scaling by optimizing goodness of fit to a nonmetric
#     hypothesis" Kruskal, J. Psychometrika, 29, (1964)
#
#     """
#     def __init__(self, depths, alpha, n_components=2, metric=True, n_init=4,
#                  max_iter=300, verbose=2, eps=1e-3, n_jobs=1,
#                  random_state=None, dissimilarity="euclidean"):
#         self.n_components = n_components
#         self.dissimilarity = dissimilarity
#         self.metric = metric
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.eps = eps
#         self.verbose = verbose
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.depths = depths
#         self.alpha = alpha
#
#     @property
#     def _pairwise(self):
#         return self.kernel == "precomputed"
#
#     def fit(self, X, init=None, y=None):
#         """
#         Computes the position of the points in the embedding space
#
#         Parameters
#         ----------
#         X : array, shape=[n_samples, n_features]
#             Input data.
#
#         init : {None or ndarray, shape (n_samples,)}, optional
#             If None, randomly chooses the initial configuration
#             if ndarray, initialize the SMACOF algorithm with this array.
#         """
#         self.fit_transform(X, init=init)
#         return self
#
#     def fit_transform(self, X, init=None, y=None):
#         """
#         Fit the data from X, and returns the embedded coordinates
#
#         Parameters
#         ----------
#         X : array, shape=[n_samples, n_features]
#             Input data.
#
#         init : {None or ndarray, shape (n_samples,)}, optional
#             If None, randomly chooses the initial configuration
#             if ndarray, initialize the SMACOF algorithm with this array.
#
#         """
#         if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
#             warnings.warn("The MDS API has changed. ``fit`` now constructs an"
#                           "dissimilarity matrix from data. To use a custom "
#                           "dissimilarity matrix, set "
#                           "``dissimilarity=precomputed``.")
#
#         if self.dissimilarity is "precomputed":
#             self.dissimilarity_matrix_ = X
#         elif self.dissimilarity is "euclidean":
#             self.dissimilarity_matrix_ = euclidean_distances(X)
#         else:
#             raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
#                              " Got %s instead" % str(self.dissimilarity))
#
#         self.embedding_, self.stress_ = smacof(self.depths, self.alpha,
#             self.dissimilarity_matrix_, metric=self.metric,
#             n_components=self.n_components, init=init, n_init=self.n_init,
#             n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
#             eps=self.eps, random_state=self.random_state)
#
#         return self.embedding_
