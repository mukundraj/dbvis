'''
Created on Nov 11, 2016

Module for functions performing math operations.

@author: mukundraj
'''

import numpy as np
from produtils import dprint

from vectors import *

# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    mtv = vector(nearest, start)
    return (dist, mtv)


def get_minkowski_diff(A_pt, B_cset):
    """

    Args:
        s1: Convex set 1
        s2: Convex set 2

    Returns:
        min_diff: Minkowski difference set
    """
    xs = B_cset['xs']
    ys = B_cset['ys']

    x = A_pt[0]
    y = A_pt[1]

    min_diff = np.zeros((len(xs), 2))

    for i in range(len(xs)):

        min_diff[i] = np.array([x-xs[i], y-ys[i]])



    return min_diff


def get_min_penetration_vector(a, mink_diff):
    """Get the minimum translation vector.

    Args:
        A_pt: Point from which to consider dist
        B_cset: The convex set
    Returns:
        mpv: minimum penetration vector
    """

    mindis = np.Inf
    mpv = None
    for i in range(len(mink_diff)-1):

        dist, nearest = pnt2line(a, mink_diff[i], mink_diff[i+1])
        # dprint(dist, mindis, dist<mindis)
        if dist<mindis:
            mpv = nearest

    return mpv