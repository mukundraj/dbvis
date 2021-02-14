"""
Created on 2017-11-25

Helper functions for visualization.


@author: mukundraj

"""

import numpy as np
from produtils import dprint
import matplotlib.pyplot as plt



filename = 'chem_sp_vectors.txt'
outpath = 'output_tsvs/vector_figs/'

def draw_hist_figures(filename, outpath):
    """
    Main file for chem mutag networks/ fb nets shortest path kernels for getting the
    vector histogram images to view with the interactive visualization.
    Args:
        filename:
        outpath:

    Returns:

    """

    # read text file
    X = np.genfromtxt(filename)

    maxval = np.amax(X)
    ymax = maxval*1.1

    m,n = np.shape(X)
    bins = range(n)
    plt.figure()

    for i in range(m):

        # get row
        row = X[i,:]
        plt.plot(row)

        outfile = format(i, '04')+'.png'
        plt.ylim(ymax = ymax, ymin = 0)

        plt.savefig(outpath+outfile)

        plt.clf()

    plt.close()
