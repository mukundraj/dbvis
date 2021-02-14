'''
Created on Aug 24, 2015

@author: mukundraj
'''
import pybel
import openbabel as ob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def mol_to_networkxgraph(mol):
    """Converts open babel molecule to networkx graph.
    
    Ref1: http://baoilleach.blogspot.com/2008/10/molecular-graph-ics-with-pybel.html
    Ref2: http://openbabel.org/docs/dev/UseTheLibrary/PythonExamples.html
    
    
    """
    
    edges = []
    bondorders = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        bondorders.append(bond.GetBO())
        edges.append( (bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1) )
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def flatten_3d_to_2d(A_mats):
    """Converts stack of adj mats to stack of flattened adj mats.

    Args:
        A_mat (3D numpy array): Stack of adjacency mats

    Returns:
        A_vecs (2D numpy array): Stack of vectors

    """

    shape = A_mats.shape
    A_vecs = np.zeros(shape=(shape[0]*shape[1], shape[2]))

    for i in range(shape[2]):
        A_vecs[:,i] = A_mats[:,:,i].flatten()

    return A_vecs

## Test code mol_to_networkxgraph

# filename = "/Users/mukundraj/Desktop/work/projects/graphdepth/data/2015-08-16/chemicals3/TR131b.sdf"
# 
# mol = pybel.readfile("sdf", filename)
# mol2 = mol.next()
# G = mol_to_networkxgraph(mol2)
# 
# nx.draw_spring(G)
# plt.show()

# # Test code flatten_3d_to_2d
# c,r = np.meshgrid(range(3),range(4))
# posmat = np.dstack((r,c))
# A_vecs = flatten_3d_to_2d(posmat)
# print A_vecs.shape