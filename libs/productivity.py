'''
Created on Oct 31, 2013

@author: mraj

'''

from inspect import currentframe, getframeinfo
import inspect
import sys
import os.path

SRC_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(os.path.dirname(SRC_ROOT))+"/"
PROJ_ROOT = os.path.dirname(os.path.dirname(SRC_ROOT))+"/"

priorityDebug = False

def dprint(*arg):
    
    if  priorityDebug==False:
        callerframerecord = inspect.stack()[1]    # 0 represents this line
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        temp = info.filename.split('/')
        print arg, "##",temp[-1],info.function, info.lineno
         
    elif arg[0]=='p':
        callerframerecord = inspect.stack()[1]    # 0 represents this line
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        temp = info.filename.split('/')
        print "pdb##",arg[1:], "##",temp[-1],info.function, info.lineno
        
