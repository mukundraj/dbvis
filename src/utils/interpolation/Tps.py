"""Tps - Thin plate spline interpolation of scattered data.

"""
import numpy as np
import numpy.matlib
import copy


class Tps():


    def __init__(self, xs, ys, zs):
        """

        Args:
            xs:
            ys:
            zs:

        Returns:

        """
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.zs = np.array(zs)
        self.N = len(self.xs)

        N = self.N
        # Sets wW
        wL = self._computeWl()
        wY = np.vstack((self.zs.reshape((N,1)),np.zeros((3,1))))

        self.wW = np.linalg.solve(wL,wY)


    def _computeWl(self):
        """


        Returns:

        """
        N = self.N
        # dprint(self.xs.reshape((N,1)))
        # dprint(self.xs.T)

        rx = np.matlib.repmat(self.xs.reshape((N,1)),1,N)
        ry = np.matlib.repmat(self.ys.reshape((N,1)),1,N)

        wR = np.sqrt((rx-rx.T)**2 + (ry-ry.T)**2)
        wP = np.hstack((np.ones((N,1)), self.xs.reshape((N,1)),self.ys.reshape((N,1))))


        wK = self._phi(wR)

        wL = np.vstack((np.hstack((wK,wP)),np.hstack((wP.T,np.zeros((3,3))))))
        # dprint(np.shape(rx),np.shape(ry),np.shape(wR),np.shape(wP),np.shape(wK))

        return wL


    def _phi(self, r):

        r2 = copy.deepcopy(r)
        zero_inds = np.where(r==0)
        r2[zero_inds] = 10e-15 #np.finfo(np.double).tiny

        phi = (r**2)*np.log(r2**2)
        return phi

    def __call__(self, *args, **kwargs):
        """ To return the interpolated values.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        # dprint(self.wW)

        shp = args[0].shape
        N = self.N
        X,Y = np.asarray([a.flatten() for a in args], dtype=np.float_)
        NWs = len(X)

        # all points in the plane
        rX = np.matlib.repmat(X,N,1)
        rY = np.matlib.repmat(Y,N,1)

        # landmark points
        xp = self.xs.reshape((N,1))
        yp = self.ys.reshape((N,1))
        rxp = np.matlib.repmat(xp,1,NWs)
        ryp = np.matlib.repmat(yp,1,NWs)

        # dprint(np.shape(rX),np.shape(rY),np.shape(rxp),np.shape(ryp))

        # mapping
        wR = np.sqrt((rxp-rX)**2 + (ryp-rY)**2)
        wK = self._phi(wR)
        wP = np.hstack((np.ones((NWs,1)), X.reshape((NWs,1)), Y.reshape((NWs,1)))).T
        wL = np.vstack((wK,wP)).T
        # dprint(np.shape(wP),np.shape(wL))

        Xw = np.dot(wL,self.wW)
        Xw = Xw.reshape(shp)
        #np.dot(self._function(r), self.nodes).reshape(shp)

        return Xw


