import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = np.genfromtxt('../extern/r/fitmono/datax.csv', delimiter=',',skip_header=1)
x = x[:,1]
y = np.genfromtxt('../extern/r/fitmono/datay.csv', delimiter=',',skip_header=1)
y = y[:,1]
z = np.genfromtxt('../extern/r/fitmono/dataz.csv', delimiter=',',skip_header=1)
Z = z[:,1:]


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# for i in range(len(x)):
#     # points2d[i] = points2d[i] - median
#     x[i],y[i] = pol2cart(x[i],y[i])

Z = Z.T
print(np.shape(Z))
X,Y = np.meshgrid(x,y)


# X,Y = pol2cart(X,Y)

plt.figure()
CS = plt.contour(X, -Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()

# http://stackoverflow.com/questions/20924085/python-conversion-between-coordinates