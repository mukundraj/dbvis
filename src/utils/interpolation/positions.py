"""
Created on 2016-12-09

Function/s to find positions of the points on a grid.

@author: mukundraj

"""
import numpy as np
from produtils import dprint
import src.datarelated.processing.misc as misc
from src.utils.interpolation.Tps import Tps
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import copy

def get_pos_on_grid(cur_pos, res):
    """Gets the index positions on the grid based on the positions of the points passed.
    Assumes a 1.5 factor expansion of the range of the positions.
    Grid is constructed using the passed 'res' parameter.

    Args:
        cur_pos: current point positions.
        res: resolution.
    Returns:
        pos_ids (list): a list of indices on the grid corresponding to point positions.
    """
    pos_ids = []

    xs = [pt[0] for pt in cur_pos]
    ys = [pt[1] for pt in cur_pos]

    maxx = max(xs)
    minx = min(xs)
    maxy = max(ys)
    miny = min(ys)

    rx = maxx - minx
    ry = maxy - miny

    max_x = maxx+0.25*rx
    min_x = minx-0.25*rx
    max_y = maxy+0.25*ry
    min_y = miny-0.25*ry


    range_x = max_x - min_x
    range_y = max_y - min_y

    dx = range_x/res
    dy = range_y/res


    xi = np.linspace(min(xs)-0.25*rx, max(xs)+0.25*rx, res)
    yi = np.linspace(min(ys)-0.25*rx, max(ys)+0.25*ry, res)

    x_inds = np.round((xs-min_x)/dx)
    y_inds = np.round((ys-min_y)/dy)


    for i in range(len(cur_pos)):
        pos_ids.append(np.array([int(x_inds[i]),int(y_inds[i])]))

    return pos_ids


def get_pos_on_grid2(cur_pos, res,XI2,YI2):
    """Gets the index positions on the grid based on the positions of the points passed.
    *NO* assumption of 1.5 factor expansion of the range of the positions.
    Grid is constructed using the passed 'res' parameter.

    Args:
        cur_pos: current point positions.
        res: resolution.
        XI2 (2d array):
        YI2 (2d array):
    Returns:
        pos_ids (list): a list of indices on the grid corresponding to point positions.
    """
    pos_ids = []

    xs = [pt[0] for pt in cur_pos]
    ys = [pt[1] for pt in cur_pos]

    if XI2 is not None and YI2 is not None:
        maxx = XI2[-1,-1]
        minx = XI2[0,0]
        maxy = YI2[-1,-1]
        miny = YI2[0,0]
    else:
        assert(False)

    rx = maxx - minx # range
    ry = maxy - miny

    # max_x = maxx+0.25*rx
    # min_x = minx-0.25*rx
    # max_y = maxy+0.25*ry
    # min_y = miny-0.25*ry

    # range_x = max_x - min_x
    # range_y = max_y - min_y

    dx = rx/(res-1) # a real dis per interval
    dy = ry/(res-1)

    # xi = np.linspace(min(xs)-0.25*rx, max(xs)+0.25*rx, res)
    # yi = np.linspace(min(ys)-0.25*rx, max(ys)+0.25*ry, res)

    x_inds = np.round((xs-minx)/dx)
    y_inds = np.round((ys-miny)/dy)


    for i in range(len(cur_pos)):
        pos_ids.append(np.array([int(x_inds[i]),int(y_inds[i])]))

    # exit(0)
    return pos_ids


def interpolate_depth_vals(cur_pos, depths, res, imname=None):
    """Interpolates the depth values on a plane with resolution of res.

    Args:
        pos:
        depths:
        res:
        filename: filename to write the interpolated plane as a figure.

    Returns:

    """
    cur_pos = copy.deepcopy(cur_pos)
    sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])

    ori_median_ind = sorted_inds[-1]
    median_pos = copy.deepcopy(cur_pos[sorted_inds[-1]])

    max_dist_from_median = 0
    for i in range(len(cur_pos)):
        cur_pos[i] -= median_pos

    dprint(cur_pos)
    dprint(ori_median_ind)
    # compute the tps spline
    xs = [pt[0] for pt in cur_pos]
    ys = [pt[1] for pt in cur_pos]

    tps = Tps(xs, ys, depths)
    # tps = Tps(xs, ys, depths_curpos)

    # polarize
    phis = np.linspace(-np.pi, np.pi, res)
    # rhos = np.linspace(0, max_dist_from_median*1.2 , self.res)
    rhos = np.linspace(0, 1, res)

    PI, RI = np.meshgrid(phis, rhos)
    ZI_pol = misc.get_depths_for_polar_grid(PI,RI,tps)

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

    ZI_pol_flat = ZI_pol.flatten()
    ZI_pol_flat = ZI_pol_flat.reshape(np.shape(X_flat))

    ZI_car_mono = griddata(np.hstack((X_flat,Y_flat)), ZI_pol_flat, (XI2, YI2), method='cubic')
    ZI_car_mono = ZI_car_mono.reshape((res,res))
    ZIcm = np.nan_to_num(ZI_car_mono) # just making the name shorter for convinience.

    if imname:

        ZIcm_max = np.amax(ZIcm)
        ZIcm_min = np.amin(ZIcm)
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

        # imname ='output_tsvs/spline_only_'+format(int(math.floor(iter/spline_lag)), '04')+'.png'
        fig.savefig(imname, dpi=100)
        im = Image.open(imname)
        im.putalpha(192)
        im.save(imname)
        plt.close()

    return cur_pos