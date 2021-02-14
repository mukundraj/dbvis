# test code for testing external (C++ exact half space depth) Dec 28, 2016


import numpy as np
from produtils import dprint
import src.analysis.halfspace_analyzer as hsa
import matplotlib.pyplot as plt
import ctypes

import produtils

x = [1,-1,1,0]
y = [1,-1,0,0]

# Get points
points2d = []
for i in range(len(x)):
    points2d.append(np.array([x[i],y[i]]))


points2d = np.array(points2d)
N,d = np.shape(points2d)
x = []
for i in range(d):
    x = x+list(points2d[:,i])


dprint(N,d, x)


# Get depths in the original space using kernel half space depth
analyser = hsa.analyzer(points2d, grid_members=None, kernel=None) # gridpoints2d is vestigial here
depths = analyser.get_depths_extern() # allproj is vestigial here

dprint(depths)



# _sum = ctypes.CDLL('../extern/cpp/HD.so')
# _sum.our_function.argtypes = (ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.c_double))
# _sum.our_function.restype = ctypes.c_double
#
# def get_HD_for_x(N, d, x):
#     """
#
#     Args:
#         N: Number of points.
#         d: Dimension of space.
#         x (list): A list composed of all dimensions stacked in order.
#
#     Returns:
#
#     """
#
#     array_type = ctypes.c_double * (N * d)
#
#     result = _sum.our_function(ctypes.c_int(N), ctypes.c_int(d), array_type(*x))
#     return float(result)
#
# dprint(get_HD_for_x(N=4, d=2, x=x))