'''
Created on Jan 15, 2017

Class for computing spatial depth using gram matrix.

@author: mukundraj
'''
# Python package for computing depths??

import numpy as np
from produtils import dprint
import src.datarelated.processing.depths as dp


class SpatialDepth:

    def __init__(self, G):
        """

        Returns:

        """

        self.G = G
        self.deltas = self.compute_deltas()

    def compute_deltas(self):
        """Computes and stores the delta matrix

        Args:
            xid: The index of 'x' whose depth is being computed.
        Returns:

        """
        m,n = np.shape(self.G)
        assert(m==n)
        deltas = np.zeros((m,n))

        for i in range(m):
            for j in range(n):
                deltas[i,j] = self.G[i,i] + self.G[j,j] - 2*self.G[i,j]

        deltas[deltas<0] = 0
        deltas = np.sqrt(deltas)



        return deltas

    def get_depths_from_gram(self):
        """

        Returns:

        """
        m,n = np.shape(self.G)
        depths = np.zeros(m)

        for xid in range(m):
            for yid in range(m):
                for zid in range(m):
                    if xid==yid or xid==zid:
                        N = 0

                    else:
                        D = self.deltas[xid,yid]*self.deltas[xid,zid]
                        if D != 0:
                            N = (self.G[xid,xid] + self.G[yid,zid] - self.G[xid,yid] - self.G[xid, zid])/D
                        else:
                            N = 0
                        # dprint(N,xid,yid,zid, (self.deltas[xid,yid]*self.deltas[xid,zid]))
                        # dprint((self.deltas[xid,yid]*self.deltas[xid,zid]),xid,yid,zid)
                    depths[xid] = depths[xid] + N
                # depths[xid] = depths[xid]/(self.deltas[xid,yid]*self.deltas[xid,zid])
                # dprint((self.deltas[xid,yid]*self.deltas[xid,zid]))
            depths[xid] = 1-np.sqrt(depths[xid])/(m-1)

        # depths = dp.histeq(np.array(depths))
        return depths
