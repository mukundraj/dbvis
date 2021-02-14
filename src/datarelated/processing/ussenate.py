'''
Created on Sep 13, 2015

@author: mukundraj

Module containing functions to process the US senate data.
'''
import glob
from libs.productivity import *
import csv
import numpy as np
from scipy import stats
import networkx as nx

def get_graph(inputfolder, year, outputfolder = None):
    """Generates a graph from the us senate voting data csv.
    
    First reads all the csvs for a particular year and aggregates the data. 
    Next computes a correlation matrix and uses the correlation values as 
    weights.
    
    Args:
        inputfolder: Read input csvs from this folder
        year: The year to be used for generating the graph
        outputfolder: Optional. If give, then writes graph in gml format to
        this folder.
        
    Returns:
        G: The graph based on correlation.
    
    """
    
    # Each key in this dictionary represents a node in the graph. Its a state code + party code
    # combination.
    ddict = {}
    
    ###############################################
    ## Read all the data into the dictionary. 
    ###############################################
    filenames = glob.glob(inputfolder+'/*_'+str(year)+'_*.csv')
    yeanay = {"Yea":1, "Nay":-1, "Not Voting":0, "Present":0,
              "Not Guilty": 0, "Guilty": 1}
    
    
    with open(filenames[0], 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                ddict[(row[1],row[5][0])] = []
    
    
    for i,filename in enumerate(filenames):
        with open(filename, 'rU') as f:
            
            reader = csv.reader(f)
           
            for row in reader:
                #print row
                try:
                    if len(ddict[(row[1],row[5][0])]) < i+1:
                        ddict[(row[1],row[5][0])] = ddict[(row[1],row[5][0])] + [yeanay[row[3]]]
                    else:
                    
                        ddict[(row[1],row[5][0])][i] = ddict[(row[1],row[5][0])][i] + yeanay[row[3]]
            
                except KeyError:
    
                    dprint("missing key: ", (row[1],row[5][0]))
                    ddict[(row[1],row[5][0])] = [] + [yeanay[row[3]]]
    
    ###############################################
    ## Next compute a correlation matrix.
    ###############################################
    
    # First compute the median length. Consider on votes by senetors 
    # who have voted the median number of times.n
    lengths = [len(ddict[key]) for key in ddict.keys()]
    median_len = np.median(lengths)
    num_median = lengths.count(median_len)
    
    data_mat = np.zeros([median_len, num_median])
    
    node_labels = {}
    idx = 0;
    for key in ddict.keys():
        if (len(ddict[key]) == median_len):
            data_mat[:,idx] = ddict[key] 
            node_labels[idx] = str(key)
            idx = idx + 1

    
    
    ###############################################
    ## Convert correlation matrix to networkx graph.
    ###############################################
    rho, pval = stats.spearmanr(data_mat)
    
    G = nx.Graph(rho)
    nx.set_node_attributes(G, 'label', node_labels)

    if outputfolder != None:
        nx.write_gml(G, outputfolder+'/'+str(year)+'.gml')
    


#################################################
## Test code. Comment the following after testing.
#################################################
# inputfolder = "/Users/mukundraj/Desktop/work/projects/graphdepth/data/2015-09-13/us_senate"
# outputfolder = "/Users/mukundraj/Desktop/work/projects/graphdepth/data/processed/us_senate"
# 
# #get_graph(inputfolder , 1999, outputfolder)
# 
# for year in range(1990,2016):
#     dprint(year)
#     get_graph(inputfolder , year, outputfolder)


