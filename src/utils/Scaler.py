"""
Created on 2016-12-04

A class to deal with scaling.

@author: mukundraj

"""
import numpy as np

class Scaler:

    def __init__(self, xs):

        self.minx = np.amin(xs)
        self.maxx = np.amax(xs)
        self.range = self.maxx - self.minx

    def to_unit(self,x):
        """ Converts to unit scale

        Returns:

        """
        units = (x - self.minx)/self.range

        return units


    def fr_unit(self,units):
        """ Converts from unit scale

        Returns:

        """

        x = units*self.range + self.minx
        return x



# x = np.linspace(10,40,100)
# scl = Scaler(x)
#
# ans=   scl.to_unit(np.linspace(10,20,100))
# print scl.fr_unit(ans)