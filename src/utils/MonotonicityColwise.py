"""
Created on 2016-12-08

Class to monotonize 2d scalar field along a single axis. This class is for doing this
column wise and keeping the other axis independent.

@author: mukundraj

"""

import numpy as np
from produtils import dprint
from src.utils.Scaler import Scaler
from sklearn.neighbors import KernelDensity
from scipy import interpolate


class MonotonicityColwise:

    def __init__(self,Y,Z):
        """

        Args:

            Y: The supposed x coordinates
            Z: The f(x) values

        Returns:

        """

        self.Y = Y
        self.Z = Z


    def __call__(self,bw):
        """

        Returns:
            monofit:

        """

        m,n = np.shape(self.Z)
        res = m
        k_grid = np.empty_like(self.Z)

        z_grid = np.linspace(0,1,res)

        # Iterate over cols (the isoval axes of X)
        for i in range(n):
            zs = self.Z[:,i]
            ys = self.Y[:,i]
            z_scaler = Scaler(zs)
            y_scaler = Scaler(ys)

            y = y_scaler.to_unit(ys)
            z = z_scaler.to_unit(zs)

            za = np.linspace(0,1,res)

            z = z.reshape(res,1)
            za = za.reshape(res,1)

            kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(z)
            kde2 = np.exp(kde.score_samples(za))
            kde2 = kde2/np.sum(kde2) # normalize for the grid size

            kde2 = kde2[::-1]
            g = np.cumsum(kde2)
            g = g[::-1]

            get_new_z_at = interpolate.interp1d(g, y, bounds_error=False, fill_value='extrapolate',kind='linear')


            new_zs = get_new_z_at(z_grid)
            new_zs = z_scaler.fr_unit(new_zs)
            # g2 = y_scaler.fr_unit(g)

            k_grid[:,i] = new_zs

        return k_grid
