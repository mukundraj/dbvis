"""
Created on 2016-11-30

Testing for my TPS (thin plate spline) code.

@author: mukundraj

"""
import numpy as np
import src.analysis.kernels as kers
from produtils import dprint
from scipy.interpolate import Rbf
from src.utils.interpolation.Tps import Tps
import matplotlib.pyplot as plt
from matplotlib import cm
import src.datarelated.processing.dim_reduction as dr
import src.analysis.halfspace_analyzer as hsa
import src.datarelated.readwrite.d3_related as d3r

np.random.seed(19)
p = {
    'kernel':'linearkernel',
    'alpha':1.5,
    'tsv_filename':"output.tsv",
    'json_filename':'output.json',
    'tsv_band_filename':'output_band.tsv'
}

depths_dict = {}

N = 3
x = np.random.rand(N)*4.0-2.0
y = np.random.rand(N)*4.0-2.0



points2d = []
for i in range(len(x)):
    points2d.append(np.array([x[i],y[i]]))

points2d.append(np.array([-0.5,-1]))
depths = np.array([-10,10,0,5])
sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])
sorted_depths = depths[sorted_inds]

cur_pos = points2d
xs = [pos[0] for j,pos in enumerate(cur_pos)]
ys = [pos[1] for j,pos in enumerate(cur_pos)]
xi = np.linspace(min(xs)-1, max(xs)+1, 100)
yi = np.linspace(min(ys)-1, max(ys)+1, 100)


rbf = Rbf(xs, ys, depths, function='thin-plate')
XI, YI = np.meshgrid(xi, yi)
ZI = rbf(XI, YI)

tps = Tps(xs, ys, depths)
ZI_tps = tps(XI, YI)

plt.figure()
plt.subplot(1, 2, 1)
plt.axis('equal')
plt.title('TPS scipy')
plt.pcolor(XI, YI, ZI, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(ZI), vmax=np.amax(ZI))
plt.colorbar()
cp = plt.contour(XI, YI, ZI, levels=sorted_depths)
plt.clabel(cp, inline=True,fontsize=10)
ax = plt.gca()
for i2 in range(len(xs)):
    ax.annotate(str(i2), (xs[i2],ys[i2]))

dprint(sorted_depths)
plt.subplot(1, 2, 2)
plt.axis('equal')
plt.title('TPS my')
plt.pcolor(XI, YI, ZI_tps, cmap=cm.jet, vmin=np.amin(ZI_tps), vmax=np.amax(ZI_tps))
plt.scatter(xs, ys, 100, depths, cmap=cm.jet, vmin=np.amin(ZI_tps), vmax=np.amax(ZI_tps))
plt.colorbar()
cp = plt.contour(XI, YI, ZI_tps, levels=sorted_depths, colors='k')
plt.clabel(cp, inline=True,fontsize=10)
ax = plt.gca()
for i2 in range(len(xs)):
    ax.annotate(str(i2), (xs[i2],ys[i2]))

plt.show()



dprint('done')