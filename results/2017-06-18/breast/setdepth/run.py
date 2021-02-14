"""
Created on 2017-11-26

Main file to experiment with breast cancer dataset and set depth.

@author: mukundraj

"""

import src.datarelated.readwrite.datacsvs as dc
from produtils import dprint
import src.analysis.kcat.kernels.functions as kfs
import src.analysis.set_analyzer as sna
import numpy as np
import src.analysis.set_analyzer as sna
import src.analysis.kernels as kers
import src.datarelated.processing.dim_reduction as dr
import src.visualization.garl as garl
import src.datarelated.readwrite.datacsvs as rw


infile = '/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-10-12/breast/breast-cancer.data'

X = dc.get_breast_data(infile, clas=2)
dc.write_csv_for_breast_vis(infile, clas=2)
dprint(X)

dprint(np.shape(X))


G = kfs.k0(X,X)


dprint(np.shape(G))

kernels = kers.kernels(X=None, kernelname=None, G=G)
similarities = kernels.get_distance_from_precomputed_gram()


depths_dict = {}
gridpoints2d = None

points2d_mds = dr.get_nmds_projections(data_list=None, similarities=similarities)

analyzer = sna.analyzer(X,2)

depths = analyzer.get_set_depth()

depths = np.array(depths)
dprint(depths)
depths = depths - np.float(np.min(depths))
depths = depths/np.float(np.max(depths))
dprint(depths)

np.random.seed(15)
p = {
        'N':200,
        'alpha':1.5,
        'pol_res':200,
        'mono_bw':0.01, # kde bandwidth in monofit 0.0001
        'N_i': 401, # number of iterations
        'M_i':0, # number of constrained iterations
        'show_fig': False, # show or write figures
        'max_step': 0.02,
        'alp':1, # multiplying coefficient for depth forces
        'spline_lag':25, # num of iters to skip before recomputing the spline
        'num_old': 1, # num of old positions to remember for time averaging
        'num_old_spline':2, # num of old splines to remember
        'fixed_stepsize':0.01,#0.01,#0.01, ##!!!IMPORTANT!!! change this to 1 or 2 for switching bw brandes/arl
        'depth_factor':5, #factor to multiply depth forces by ##!!!IMPORTANT!!! change this for switching bw brandes/arl
        'smooth_factor':0.0, #factor to multiply smooth forces by
        'depth_type': 1, # 0: halfspace , 1: spatial
        'alpha': 1.5
}

N = len(points2d_mds)
A = np.zeros((N,2))

garl = garl.garl(depths, p['N_i'],p['M_i'],res=p['pol_res'], bw=p['mono_bw'], max_step=p['max_step'],
                       num_old=p['num_old'], num_old_spline=p['num_old_spline'], depth_type=p['depth_type'],  alpha=p['alpha'])
points_dmds = garl.get_mds_dc(None, points2d_mds, p['show_fig'], p['spline_lag'], p['depth_factor'], p['smooth_factor'],
                              similarities, p['fixed_stepsize'])
# rw.write_distances_tsv(points_dmds, "dists_dmds.tsv", points_dmds)

dprint("done")
