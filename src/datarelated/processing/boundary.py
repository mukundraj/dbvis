"""
Created on 2016-12-02

Functions to generate boundary points for TPS interpolation of depths.

@author: mukundraj

"""

import numpy as np
from produtils import dprint

import scipy.spatial.distance as distance
from shapely.geometry import MultiPoint

def get_boundary_points(points2d, depths):
    """

    Args:
        points2d:
        depths:

    Returns:
        new_points2d(list): List of points with boundary points appended.
        new_depths(list): List of depths with the depths of boundary points appended.
        max_dist: max_distance from median
    """

    sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])
    median = sorted_inds[-1]
    median_d = depths[median]

    max_dis = 0

    for i in range(len(depths)):
        dis = distance.euclidean(points2d[i], points2d[median])
        if dis>max_dis:
            max_dis = dis

    hull = MultiPoint(points2d).convex_hull

    hull_boundary_pts = list(hull.exterior.coords)

    all_depths = np.append(depths,np.zeros((len(hull_boundary_pts))))
    boundary_pts = []
    dprint(median_d)
    for j in range(len(hull_boundary_pts)):

        c_id = [i for i,pt in enumerate(points2d) if all(pt==hull_boundary_pts[j])][0]
        md_dis = median_d - depths[c_id]# depth distance from median's depth
        # md_frac = float(md_dis)/median_d
        md_factor = median_d/float(md_dis)

        cpos = (points2d[c_id] - points2d[median])*md_factor
        cpos = cpos + points2d[median]
        boundary_pts.append(cpos)
    dprint(len(hull_boundary_pts))

    # points2d = [np.array([points2d[i][0],points2d[i][1]]) for i in range(len(points2d))]
    # ids = [i for i,pt in enumerate(points2d) if all(pt==pts[1])]
    # dprint(ids)
    # dprint(len(pts))

    all_points2d = []
    all_points2d.extend(points2d)
    all_points2d.extend(boundary_pts)
    all_depths = np.append(depths,np.zeros((len(boundary_pts))))

    return all_points2d,all_depths,max_dis

