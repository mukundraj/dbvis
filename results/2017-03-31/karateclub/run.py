"""
Created on 2017-03-31

Main file to experiment with rmds on karage club data.

@author: mukundraj

"""

from produtils import dprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import src.visualization.rmds as rmds
# import src.visualization.brandes as rmds

import src.datarelated.processing.dim_reduction as dr
import src.datarelated.readwrite.datacsvs as rw
import networkx as nx

np.random.seed(15)
p = {
        'N':200,
        'alpha':1.5,
        'pol_res':200,
        'mono_bw':0.01, # kde bandwidth in monofit 0.0001
        'N_i':501 , # number of iterations
        'M_i':0, # number of constrained iterations
        'show_fig': False, # show or write figures
        'max_step': 0.02,
        'alp':1, # multiplying coefficient for depth forces
        'spline_lag':50, # num of iters to skip before recomputing the spline
        'num_old': 5, # num of old positions to remember for time averaging
        'num_old_spline':2, # num of old splines to remember
        'fixed_stepsize':1,
        'depth_factor':3, #factor to multiply depth forces by
        'smooth_factor':0.0, #factor to multiply smooth forces by
        'depth_type': 1 # 0: halfspace , 1: spatial
}

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G)
L = nx.laplacian_matrix(G)
d = nx.diameter(G)
node_names = [G.nodes(data=True)[x][1]['club'] for x in range(len(G.nodes()))]
rw.write_node_names_to_json(node_names)


ds = nx.betweenness_centrality(G)
# ds = nx.closeness_centrality(G)

depths = []
n = len(ds)
for i in range(n):
    depths.append(ds[i])

depths = np.array(depths)
depths = depths - np.min(depths)
depths = depths/np.max(depths)


paths=nx.all_pairs_shortest_path_length(G)

similarities = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        similarities[i,j] = paths[i][j]
similarities = similarities/np.amax(similarities)

points2d_mds = dr.get_nmds_projections(data_list=None, similarities=similarities, metric_mds=True)
# dprint(points2d_mds)

# points2d_mds = []
# points = np.random.rand(len(depths),2)
# dprint(np.shape(points))
#
# for i in range(len(depths)):
#     points2d_mds.append(np.array(points[i,:]))



# # mds
# xs = [x[0] for x in points2d_mds]
# ys = [x[1] for x in points2d_mds]
# # plt.plot(xs, ys, 'ro')
# plt.scatter(xs, ys, 30, depths, cmap=cm.jet, vmin=np.amin(depths), vmax=np.amax(depths))
# # plt.scatter(xs, ys, 20)
# plt.colorbar()
# ax = plt.gca()
# for i2 in range(len(xs)):
#     ax.annotate(str(i2), (xs[i2],ys[i2]))
# plt.show()
# exit(0)

rw.write_edges_tsv(G.edges(), "edges.tsv")

rmds = rmds.rmds(depths, p['N_i'],p['M_i'],res=p['pol_res'], bw=p['mono_bw'], max_step=p['max_step'],
                       num_old=p['num_old'], num_old_spline=p['num_old_spline'], depth_type=p['depth_type'], adj_mat=L)
points_dmds = rmds.get_mds_dc(None, points2d_mds, p['show_fig'], p['spline_lag'], p['depth_factor'], p['smooth_factor'],
                              similarities, p['fixed_stepsize'])
rw.write_distances_tsv(points_dmds, "dists_dmds.tsv", points_dmds)
