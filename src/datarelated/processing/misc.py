'''
Created on Nov 27, 2016

@author: mukundraj

Misc functions not classified as part of other modules.
'''

from datarelated.processing.depths import get_median_and_bands
import scipy.spatial.distance as distance
import numpy as np
from produtils import dprint

def get_max_distance_from_median(points2d, depths, alpha):
    """Gets the maximum euclidean distance from median.

    Args:
        points2d:
        depths:
        alpha:

    Returns:
        max_dis
        median

    """

    max_dis = 0

    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths, alpha=alpha)


    for i in range(len(depths)):
        dis = distance.euclidean(points2d[i], points2d[median])
        if dis>max_dis:
            max_dis = dis

    return max_dis,median

def get_boundary_points(median_xy, max_dis, res_N=10):
    """

    Args:
        median_xy:
        max_dis:
        res:

    Returns:

    """

    mx = median_xy[0]
    my = median_xy[1]

    xs = np.linspace(mx-max_dis, mx+max_dis, res_N, endpoint=True)
    ys = np.linspace(my-max_dis, my+max_dis, res_N, endpoint=True)

    boundary_points = []
    for x in xs:
            boundary_points.append(np.array([x,ys[0]]))
            boundary_points.append(np.array([x,ys[-1]]))

    for y in ys[1:-1]:
            boundary_points.append(np.array([xs[0],y]))
            boundary_points.append(np.array([xs[-1],y]))

    dprint(len(boundary_points))
    return boundary_points

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def get_depths_for_polar_grid(PI,RI,tps):
    """Get the depth values for the points on a polar grid.

    Returns:
        ZI_pol(2d array): 2D array with the depth values computed
            using the tps spline.
    """
    #
    # ZI_pol = np.empty_like(PI)
    #
    # m,n = np.shape(PI)
    #
    # for i in range(m):
    #     for j in range(n):
    #         x,y = pol2cart(RI[i,j],PI[i,j])
    #         ZI_pol[i,j] = tps(x,y)

    vfunc = np.vectorize(pol2cart)
    XI,YI = vfunc(RI,PI)
    ZI_pol = tps(XI,YI)

    return ZI_pol




def get_color(i):
    """Returns a boxplot color based on i
    var colors = ["#AAAA00","#0000ff","#00ccff","red"]
    Args:
        i:

    Returns:

    """
    if i == 0:
        return '#AAAA00'
    elif i==1:
        return '#0000ff'
    elif i==2:
        return '#00ccff'
    elif i==3:
        return '#ff0000'
    elif i==4:
        return '#000000'


