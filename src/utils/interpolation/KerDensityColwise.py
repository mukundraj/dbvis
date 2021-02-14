"""
Created on 2016-12-07

Class to do Kernel density estimation with cyclic boundary along one direction.
This version does it rowwise for the 2D Monotization. Z=f() here, and do for each X
independently.

@author: mukundraj

"""
import numpy as np
from sklearn.neighbors import KernelDensity
from produtils import dprint

class KerDensityColwise:

    def __init__(self, bw, Z):
        """

        Args:
            bw: bandwidth
            Z : "not" output of meshgrid


        Returns:

        """
        self.bw = bw
        self.Z = Z




    def smooth_sample(self, grid_Z):
       """

       Args:
            self:
            X: output of meshgrid
            Z: output of meshgrid

       Returns:
            k_grid: kde estimate (unnormalized) at the data points

       """
       k_grid = np.zeros(np.shape(grid_Z))


       return k_grid