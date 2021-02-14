'''
Created on Jun 5, 2015

Various methods to process graphs

@author: mukundraj
'''
from datarelated.processing.depths import get_median_and_bands
from libs.productivity import *
import numpy as np
import networkx as nx
import glob
import ast
import os
 
def get_graph_list_from_A_mats(A_mats, th = None):
    ''' Gets graph list from A_mats. A_mats must be thresholded to 1/0.
    
    '''

    dprint(th)

    A_mats2 = np.copy(A_mats)
    if th is not None:
        A_mats2[np.where(A_mats >= th)] = 1
        A_mats2[np.where(A_mats < th)] = 0

    dprint(np.shape(A_mats))
    graphs_list = []

    shape = np.shape(A_mats2)



    for k in range(shape[2]):
        #dprint(np.count_nonzero(A_mats[:,:,k]))
        G = nx.Graph()

        for i in range(shape[0]):
            for j in range(shape[1]):
                #dprint(i,j,k, A_mats[i,j,k])
                if A_mats2[i,j,k] != 0:
                    G.add_edge(i,j)


        graphs_list.append(G)
        dprint(len(G.edges()))

    return graphs_list
        
        
        
def nxgraphlist_to_adjmatlist(graphs_list, weight_attribute = 'corval'):
    """Converts a networkx graphs in a list to 
    adjacency matrices.
        Further combines the adjacency matrices into
    a 3D numpy array.
    
    returns:
        A_mats : A 3D numpy array of stacked 2D adjacency matrices.
    
    """
    graphs_adjmats = []
    ensemble_size = len(graphs_list)
    
    ## Conversion from networkx graph to adjacency matrix
    for i in range(ensemble_size):
    
        G = graphs_list[i]
        A = np.array(nx.adjacency_matrix(G, weight = weight_attribute))
        graphs_adjmats.append(A)
    
    
    ## Now converting the list of graphs to a 3D numpy array.
    A_mats = np.zeros(shape = np.shape(A)+(ensemble_size,))
    for i in range(0,ensemble_size):
        
        A_mats[:,:,i] = graphs_adjmats[i]



        
    return A_mats

def nxgraphlist_to_adjmatlist_varsize(graphs_list, names_list,
                                       output_folder = None,
                                       weight_attribute = "weight"):
    """Converts a graph ensemble to adjacency matrix ensemble when the 
    graphs have different number of vertices/ verts with different ids.
    
    Each node must have an associated label attribute for proper alignment
    of the nodes and edges in the matrix.
    
    Args:
        graph_list: list of networkx graphs , with each node having a label
            attribute.
        names_list: List of names associated with each graph. Only needed if
            writing at the moment.
        output_folder: Path to store output. Useful to store results
            to save time in future runs.
        
    
    Returns:
        A_mats: A 3D numpy array of adjacency matrices
    
    """
    
    graphs_adjmats = []
    ensemble_size = len(graphs_list)
    all_verts = {}
    
    for G in graphs_list:
        for node in G.nodes():
            all_verts[node] = None
    
    
    for i,key in enumerate(all_verts):
        all_verts[key] = i
    
    total_verts = len(all_verts)
    
    for i,G in enumerate(graphs_list):
        
        A = np.zeros([total_verts, total_verts])
        
        for edge in G.edges():
            try:
                A[all_verts[edge[0]],all_verts[edge[1]]]  = \
                                    G[edge[0]][edge[1]][weight_attribute] 

            except:
                dprint("EXCEPTION", edge, i, G[edge[0]][edge[1]][weight_attribute], weight_attribute)
        
        graphs_adjmats.append(A)
    
     
    ## Now converting the list of graphs to a 3D numpy array.
    A_mats = np.zeros(shape = np.shape(A)+(ensemble_size,))
    for i in range(0,ensemble_size):
        A_mats[:,:,i] = graphs_adjmats[i]
        
   
    ## Write output to temporary folder if a path is provided.
    if output_folder != None:
        for i,A in enumerate(graphs_adjmats):
            np.save(output_folder+names_list[i]+".npy", graphs_adjmats[i])
        
    return A_mats

    

def get_gmls_from_folder(inputfolder, tuplelable = True):
    """Reads gmls from a folder and returns a list of networkx graphs.
    
    Also handles tuple labels stored as string attibute if needed.
    
    Ref1: http://stackoverflow.com/questions/16533270/how-to-convert-tuple-in-string-to-tuple-object
    
    Args:
        inputpath: Path to the gmls
        tuplelabel: Signal if a tuple label is stored as a string and 
            needs handleing.
    
    Returns:
        graph_list: A list of graphs
        names_list: List of name strings corresponding to each graph.
    
    # Useful function below:
    # ast.literal_eval(s)
    """
    filenames = glob.glob(inputfolder+'/*.gml')
    
    names_list = []
    
    graphs_list = []
    
    for filename in filenames:
        nameext = os.path.basename(filename)
        name, ext = os.path.splitext(nameext)
        names_list.append(name)
        
        G = nx.read_gml(filename,  relabel = True)
        node_labels = {}
        
        for nodeid in G.nodes():
            node_labels[nodeid] = ast.literal_eval(nodeid)
            
            
        nx.relabel_nodes(G, node_labels, copy = False)
        graphs_list.append(G)
    
    return graphs_list, names_list


def get_band_degrees(depths, graphs, alpha, num_nodes):
    """Gets degrees of verts based on bands for reordering during visualization.

    Args:
        depths: Depths of graphs in ensemble
        graphs: Graph ensemble (list of graphs)

        p: All parameters ( p[alpha] Parameter for deciding depth threshold for outliers, negative
            means the number of outliers are hardcoded to abs(alpha)

    Returns:
        data_dict

    """
    alpha

    degrees_median = np.zeros(num_nodes)
    degrees_b50 = np.zeros(num_nodes)
    degrees_b100 = np.zeros(num_nodes)
    degrees_all = np.zeros(num_nodes)


    median, band50, band100, outliers, cat_list, band_prob_bounds_list = get_median_and_bands(depths, alpha=alpha)

    median_graph = nx.Graph()
    band50_graph = nx.Graph()
    band100_graph = nx.Graph()
    outlier_graph = nx.Graph()
    union_graph = nx.Graph()

    for i, G in enumerate(graphs):

        union_graph.add_edges_from(G.edges())
        # Deal with median graph edges
        if i == median:
            band100_graph.add_edges_from(G.edges())
            band50_graph.add_edges_from(G.edges())
            median_graph.add_edges_from(G.edges())




        # Deal with 50 percent band graph union graph edges
        elif i in band50:
            band100_graph.add_edges_from(G.edges())
            band50_graph.add_edges_from(G.edges())

        # Deal with 100 percent band union graph edges
        elif i in band100:
            band100_graph.add_edges_from(G.edges())

        elif i in outliers:
            outlier_graph.add_edges_from(G.edges())

    graphs = [median_graph, band50_graph, band100_graph, union_graph]
    degrees = [degrees_median, degrees_b50, degrees_b100, degrees_all]

    for i,G in enumerate(graphs):

        for j in range(num_nodes):
            if j in G:
                degrees[i][j] = G.degree(j)

    # degrees_median = median_graph.degree(range(p['num_nodes'])).values()
    # degrees_b50 = band50_graph.degree(range(p['num_nodes'])).values()
    # degrees_b100 = band100_graph.degree(range(p['num_nodes'])).values()
    # degrees_all = union_graph.degree(range(p['num_nodes'])).values()
    # dprint(median_graph.edges())

    data_dict = {}

    data_dict['numverts'] = num_nodes#len(union_graph.nodes())
    data_dict['degrees_median'] = list(degrees_median)
    data_dict['degrees_b50'] = list(degrees_b50)
    data_dict['degrees_b100'] = list(degrees_b100)
    data_dict['degrees_all'] = list(degrees_all)

    return data_dict
    

