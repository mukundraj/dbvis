'''
Created on Aug 01, 2016

Module for generating various debugging related visualitions.

@author: mukundraj
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os, sys

from libs.productivity import dprint

pi = np.pi
sin = np.sin
cos = np.cos

def show_minvol_ellipse(points, A, centroid, ax = None):
    """Draws points and also draws the surface of the ellipse.

    """

    if ax == None:
        external_ax = False;
    else:
        external_ax = True;

    U, D, V = la.svd(A)
    rx, ry, rz = 1./np.sqrt(D)
    u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

    def ellipse(u,v):
        x = rx*cos(u)*cos(v)
        y = ry*sin(u)*cos(v)
        z = rz*sin(v)
        return x,y,z

    E = np.dstack(ellipse(u,v))
    E = np.dot(E,V) + centroid
    x, y, z = np.rollaxis(E, axis = -1)

    if external_ax == False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.05)
    ax.scatter(points[:,0],points[:,1],points[:,2])

    if external_ax == False:
        plt.show()



def show_3d_rectange(center, size, ax):
    """

    Returns:

    """

    # starts

    ox, oy, oz = center
    l, w, h = size

    x = np.linspace(ox-l/2,ox+l/2,num=10)
    y = np.linspace(oy-w/2,oy+w/2,num=10)
    z = np.linspace(oz-h/2,oz+h/2,num=10)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)


    # outside surface
    ax.plot_wireframe(x1, y11, z1, color='b', rstride=1, cstride=1, alpha=0.6)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color='b', rstride=1, cstride=1, alpha=0.6)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color='b', rstride=1, cstride=1, alpha=0.6)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color='b', rstride=1, cstride=1, alpha=0.6)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6)
    ax.set_xlabel('X')
    # ax.set_xlim(-3, 3)
    ax.set_ylabel('Y')
    # ax.set_ylim(-3, 3)
    ax.set_zlabel('Z')
    # ax.set_zlim(-3, 3)

def show_3d_points_interactive(points):
    """Draws points in 3D (no ellipse surface).

    Args:
        points:

    Returns:

    """


##### TO CREATE A SERIES OF PICTURES
# Refs:
# https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/

def make_views(ax,angles,elevation=None, width=4, height = 3,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    # ax.figure.set_size_inches(width,height)

    for i,angle in enumerate(angles):

        ax.view_init(elev = elevation, azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files



##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION


def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)

    print command[output_ext]
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    command = 'convert -delay %d -loop %d %s %s'%(delay,loop," ".join(files),output)
    # os.system(command)
    dprint('Step 2: Run the above commented command manually to generate gif.')



def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s'%(" ".join(files),output))



##### MAIN FUNCTION

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax,angles, **kwargs)



    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}

    D[output_ext](files,output,**kwargs)

    dprint('Step 1: Comment out the following two lines to keep the constituent gifs.')
    for f in files:
        os.remove(f)


##### EXAMPLE

# if __name__ == '__main__':
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y, Z = axes3d.get_test_data(0.05)
#     s = ax.plot_surface(X, Y, Z, cmap=cm.jet)
#     plt.axis('off') # remove axes for visual appeal
#
#     angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
#
#     # create an animated gif (20ms between frames)
#     rotanimate(ax, angles,'movie.gif',delay=20)
#
#     # create a movie with 10 frames per seconds and 'quality' 2000
#     rotanimate(ax, angles,'movie.mp4',fps=10,bitrate=2000)
#
#     # create an ogv movie
#     rotanimate(ax, angles, 'movie.ogv',fps=10)