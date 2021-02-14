"""
Created on 2016-12-03

Class to monotonize 2d scalar field along a single axis.

@author: mukundraj

"""

import numpy as np
from sklearn.neighbors import KernelDensity
from produtils import dprint
import matplotlib.pyplot as plt
from matplotlib import cm
from src.utils.Scaler import Scaler
from src.utils.interpolation.KerDensity import KerDensity
from src.utils.interpolation.KerDensityColwise import KerDensityColwise
from scipy import interpolate


class Monotonocity:

    def __init__(self,X,Y,Z):
        """

        Args:
            Z_in:
            bandwidth:

        Returns:

        """
        self.X = X
        self.Y = Y
        self.Z = Z

        self.y_scaler = Scaler(Y)
        self.z_scaler = Scaler(Z)

        self.X = self.z_scaler.to_unit(self.X)
        self.Y = self.y_scaler.to_unit(self.Y)
        self.Z = self.z_scaler.to_unit(self.Z)




    def __call__(self, bw,res):
        """returns the monotonized version. Returns a function monotonically decreasing
        along the Y axis.

        Args:
            bw: Bandwidth

        Returns:

        """

        # kde
        xs = self.X.flatten()
        zs = self.Z.flatten()
        xzs = zip(xs,zs)

        # max_Z = np.amax(zs)
        min_X = np.amin(xs)
        max_X = np.amax(xs)



        # ## rowwise starts
        # dprint(self.X.shape,self.Z.shape)
        # dprint(self.X[0,:])
        # kd = KerDensityColwise(bw, self.Z)
        # ## rowwise ends
        #
        # exit(0)

        #old ker den start
        # kernel_sk = KernelDensity(kernel='gaussian', bandwidth=bw).fit(xzs)
        # # new ker den starts
        kd = KerDensity(bw, xzs)

        z = np.linspace(0,1,res)
        y = np.linspace(0,1,res)
        x = np.linspace(min_X, max_X,res)
        X,Z = np.meshgrid(x,z)
        xs = X.flatten()
        zs = Z.flatten()
        xzs = zip(xs,zs)
        X,Y = np.meshgrid(x,y) # creating Y grid

        # k_grid = np.exp(kernel_sk.score_samples(xzs))
        # k_grid = np.reshape(k_grid,X.shape)
        #old ker den ends
        k_grid = kd.score_sample(X,Z)
        # # new ker den ends


        # computing cdf
        k_grid =k_grid/np.sum(k_grid)
        k_grid = np.flipud(k_grid)

        k_grid = np.cumsum(k_grid, axis=0)
        k_grid = np.flipud(k_grid) #cdf

        dprint(np.shape(k_grid),np.shape(y),np.shape(x))

        # scale rows to unit axis, invert rows, scale back rows
        colmax = np.amax(k_grid,axis=0)
        k_grid = k_grid/colmax

        # Z = interpolate.interp2d(X.flatten(), k_grid.flatten(), Y.flatten(),kind='cubic')
        # rbfi = Rbf(X.flatten(), k_grid.flatten(), Y.flatten())
        for i in range(res):
            z_s = k_grid[:,i]
            y_s = Y[:,i]
            get_new_z_at = interpolate.interp1d(z_s, y_s, bounds_error=False, fill_value=1)
            #dprint(np.shape(z), np.shape(np.append(0,get_new_z_at(z[1:]))))
            # k_grid[:,i] = np.append(0,get_new_z_at(z[1:]))
            k_grid[:,i] = get_new_z_at(z)

        k_grid = k_grid*colmax


        # global scale to unit axis
        global_max = np.amax(colmax)
        k_grid = k_grid/global_max


        # scale axes back to original lengths (only need to do this for z)
        k_grid = self.z_scaler.fr_unit(k_grid)


        plt.pcolor(x, y, k_grid, cmap=cm.jet, vmin=np.amin(k_grid), vmax=np.amax(k_grid))
        plt.colorbar()
        plt.contour(x, y, k_grid, 5, colors='k', linewidths=0.6)


        monofit = k_grid

        # scale

        return monofit

