'''
Created on 21 Aug, 2015

@author: Mukund

This file has function to processes the text file with the classification for the protein
toxicology challenge data.

'''

import re
from libs.productivity import *


def process_classification_file(classfile, animalcode, testcode):
    ''' Processes raw classification file from the PTC website.
    e.g.- http://www.predictive-toxicology.org/data/ntp/corrected_results.txt
    
    parameters-
    classfile: Path to file with classification of molecules
    animalcode: PTC animal code
    testcode: PTC classification code
    
    
    returns-
    output_ids: A list of output ids as per requirement.
    
    
    '''
    
#     # Parameters
#     inputfile = '/Users/mukundraj/Desktop/work/projects/graphdepth/data/2015-08-16/chemicals3b/corrected_results.txt'
#     animalcode = 'MM' #eg : MM=male mouse
#     testcode = 'P'  # eg : P=positive for carcinogenicity
    
    inputfile = classfile
    # Read and reformat 
    with open(inputfile) as f:
        lines = f.readlines()
    
    lines = [line.rstrip('\n') for line in lines]
    
    
    final_formatted = []
    for line in lines:
        
        split_line =  re.split(" |\t",line)
        
        results = split_line[2:]
        
        result_dict = {}
        
        for pair in results:
            if pair != '':
                result_dict[pair.split('=')[0]] = pair.split('=')[1].rstrip(',')
        final_formatted.append((split_line[0],result_dict))
    
        
    # Select molecule id as per parameters.    
    output_ids = []
    for item in final_formatted:
        if animalcode in item[1].keys():
           
            
            if item[1][animalcode] == testcode:
                output_ids.append(item[0])
    
    return output_ids


