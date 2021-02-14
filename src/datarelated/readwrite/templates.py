'''
Created on 2016-05-08

Functions related to processing html templates for visualizations.

@author: mraj

'''


import shutil
import errno
import os

def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

def generate_vis_html(path_to_template, filename):
    """

    Args:
        path_to_template:

    Returns:

    """

    f = open(path_to_template+filename, 'r')
    filedata = f.read()
    f.close()

    #newdata = filedata.replace("old data","new data")

    f = open(filename,'w')
    f.write(filedata)
    f.close()

    if os.path.exists("./libs"):
        shutil.rmtree("./libs")
    copy(path_to_template+"libs", "./libs")