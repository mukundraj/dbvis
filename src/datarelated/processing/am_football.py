'''
Created on Aug 27, 2015

@author: mukundraj

Functions to process American football data.
'''

import datarelated.readwrite.am_football as amrw
#import datarelated.processing.processgraphs as pro

def get_am_football_ensemble(basefolder,year_list, teamcode_list, weight_attribute):
    """Filters and returns graph ensembles as specified.
    
    Also sets the weight as the specified attribute.
    
    Args:
        base_folder: Path to the base folder of data.
        year_list: A list of years for which to fetch data.
        teamcode_list: A list of teams for which to fetch data.
        
    Returns:
        A_mats: A 3D numpy matrix, ensemble of 2D adjacency matrices.
    
    """
    
    A_mats = None
    
    graphs_list = []
    for year in year_list:
        
        G_dict = amrw.read_am_football_data(basefolder, year, write_path = None)
        if teamcode_list != None:
            for teamcode in teamcode_list:
                
                selected_keys = [k for k in G_dict.keys() if k[1] == str(teamcode) ]
                
        else:
                selected_keys = G_dict.keys()
                
        
        for key in selected_keys:
            graphs_list.append(G_dict[key])
            
    
    
    A_mats = pro.nxgraphlist_to_adjmatlist_varsize(graphs_list, names_list = None,
                                                   output_folder = None,
                                                    weight_attribute=weight_attribute)
    
    return graphs_list
         
         

         
########
## Test Code. Comment after testing.
########

basefolder = "/Users/mukundraj/Desktop/work/projects/graphdepth/data/2015-09-13/us_football/"
get_am_football_ensemble(basefolder, [2005],teamcode_list = [28], weight_attribute = 'yards')
