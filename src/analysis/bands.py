'''
Created on May 17, 2016

Module with functions for computing various bands.

@author: mukundraj
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import math



def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1

    References:
        http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
        http://math.stackexchange.com/questions/159095/how-to-derive-ellipse-matrix-for-general-ellipse-in-homogenous-coordinates
        http://www.mathworks.com/matlabcentral/fileexchange/13844-plot-an-ellipse-in--center-form-
        https://math.stackexchange.com/questions/1227369/calculating-the-length-of-the-semi-major-axis-from-the-general-equation-of-an-el
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d
    return np.asarray(A), np.squeeze(np.asarray(c))


def get_ellipse2d_info(points):
    """Computes the parameters for drawing general ellipse using svg/d3.

    Args:
        points: A 2D numpy array Nx2

    Returns:
        ell_info (dict): {cx, cy, ra,rb, rot(in degrees)}

    """
    A,C = mvee(points)
    U, D, V = la.svd(A)
    b = A[0,1]
    a = A[0,0]
    c = A[1,1]


    tan2theta = 2*b/(a-c)
    slope = math.atan(tan2theta)/2;

    if a>c:
        slope += math.pi/2
    degslope = slope*180/math.pi
    ra, rb = [1/np.sqrt(d) for d in D]

    ell_info = {
        "cx": round(C[0],3),
        "cy": round(C[1],3),
        "ra": round(ra,3),
        "rb": round(rb,3),
        "rot": round(degslope,3)
    }

    return ell_info