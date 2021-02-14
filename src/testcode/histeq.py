# test code for histogram equalization 23 Mar 2017
# works on 1D array
import math
import numpy as np
import matplotlib.pyplot as plt
from produtils import dprint
from sklearn.neighbors import KernelDensity
from scipy import interpolate

def histmatch(a1, a2, bw=0.1, res=1000):
    """Matches the histogram of a1 to histogram on a2.

    Args:
        a1: original input historgram
        a2: the target histogram to be matched to
        bw: bandwidth
        res: the resolution for computing the cdf
    Returns:

    """

    xs = np.linspace(0,1,res)
    dprint(np.shape(a1,),np.shape(a2))
    # get T(r)
    kde_a1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a1)
    a1s = np.exp(kde_a1.score_samples(xs.reshape((res,1))))
    a1s = a1s/np.sum(a1s) # normalize for the grid size

    Tr = np.cumsum(a1s)


    # get G(z)
    kde_a2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a2)
    a2s = np.exp(kde_a2.score_samples(xs.reshape((res,1))))
    a2s = a2s/np.sum(a2s) # normalize for the grid size
    Gz = np.cumsum(a2s)


    # get G^-1(z)
    Gz_inv = interpolate.interp1d(Gz, xs, fill_value="extrapolate")

    # do the conversion and return the matched histogram
    matched = []
    for i,d in enumerate(a1):
        id = int(math.floor(d*(res-1)))
        dprint(d,id)
        matched.append(Gz_inv(Tr[id]))

    matched = np.array(matched)
    return matched

np.random.seed(10)

mu, sigma = 0.5, 0.05 # mean and standard deviation
mu2, sigma2 = 0.5, 0.2 # mean and standard deviation
s = np.random.normal(mu, sigma, 100).reshape((100,1))

s2 = np.random.normal(mu2, sigma2, 100).reshape((100,1))
hist, bin_edges = np.histogram(s, density=True)

ya = np.linspace(-1,1,1000).reshape((1000,1))
ya2 = np.linspace(0,1,1000)
# dprint(ya2)
hist = hist.reshape((10,1))

dprint(np.shape(bin_edges))

bin_edges = bin_edges.reshape((11,1))

kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(s2)
kde2 = np.exp(kde.score_samples(ya))
kde2 = kde2/np.sum(kde2) # normalize for the grid size

g = np.cumsum(kde2)
# plt.plot(ya,kde2)
# ax = plt.gca()
# ax.plot(s2[:, 0], -0.0001 * np.random.random(s2.shape[0]), '+k')
# plt.show()

f = interpolate.interp1d(g, ya2, fill_value="extrapolate")

# plt.figure()
# # plt.plot(ya,g)
# plt.plot(ya2,f(ya2))
# plt.show()

matchedhist = histmatch(s,s2, bw=0.02)
bins = np.linspace(0,1,20)
hists, bin_edges = np.histogram(s, density=True,bins=bins)
hists2, bin_edges = np.histogram(s2, density=True, bins=bins)
matchedhist, bin_edges = np.histogram(matchedhist, density=True,bins =bins)

plt.plot(hists, color='r')
plt.plot(hists2, color='g')
plt.plot(matchedhist, color='b')
plt.show()

exit(0)