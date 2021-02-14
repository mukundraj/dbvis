'''
Created on Jul 31, 2015

Module to perform functional band depth analysis.

@author: mukundraj
'''

from itertools import combinations
import numpy as np
from produtils import dprint


class anayzer_fbdmat(object):
    '''
    Class to perform functional band depth analysis on a matrix ensemble.
    '''
    
    
    def __init__(self, A_mats, r, epsilon):
        '''Constructor
        
        Args:
            A_mats: Adjacency matrices,  3D numpy array.
            r    : Number of ensemble members forming the band.
            epsilon: If negative, then generalized band depth.
                    If positive (between 0 and 1), then epsilon fraction must
                     be inside the band to be considered within the band.
        
        Returns:
            None
        '''
        
        self.A_mats = A_mats
        self.r = r
        self.epsilon = epsilon
        
    def get_combinations(self):
        '''Gets all the combinations in list
        '''
        
        mat_shape = np.shape(self.A_mats)
        combs = combinations(range(mat_shape[2]),self.r)
        
        return list(combs)
    
    
    
    def get_depths(self):
        
        '''Calculates the depth values for each of the member of the ensemble
        in self.matrices_list
        
        Inputs:
            None
        
        Returns:
            depths- list of depth values corresponding to self.matrices_list
        
        '''
        
        mats_shape = np.shape(self.A_mats)
        ensize = mats_shape[2]
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        mat_side = mats_shape[0]

        inside_combs = []
        for i in range(ensize):
            inband = 0
            per_member_combs = []
            for comb in combs:
                
                subset = self.A_mats[:,:,comb]
                # exit(0)
                cur_mat = self.A_mats[:,:,i]
                max_mat = np.amax(subset, axis = 2)
                min_mat = np.amin(subset, axis = 2)
                top_bounded = cur_mat<=max_mat
                bot_bounded = cur_mat>=min_mat
                
                inside = top_bounded & bot_bounded
                # inside = np.tril(inside,0)

                per_member_combs.append(sum(sum(inside)))
                max_possible_edges = mat_side*mat_side  # (mat_side+1)*0.5
                #if sum(sum(inside)) == mat_side*(mat_side+1)*0.5:

                if self.epsilon == 1:
                    if sum(sum(inside)) == max_possible_edges:
                        inband += 1
                elif self.epsilon == -1:
                        inband += sum(sum(inside))/float(max_possible_edges)
                else:
                    inside_frac = sum(sum(inside))/float(max_possible_edges)
                    if inside_frac >= self.epsilon:
                        inband+=1

            inside_combs.append(per_member_combs)

            depths[i] = np.sum(inband)/len(combs)
            
        return depths
    

class anayzer_fbdvec(object):
    '''
    Class to perform functional band depth analysis on a vector ensemble.
    '''


    def __init__(self, vecs_list, r, epsilon):
        '''Constructor
        '''
        
        self.vecs_list = vecs_list
        self.r = r
        self.epsilon = epsilon
        
    def get_combinations(self):
        '''Gets all the combinations in list
        '''
        combs = combinations(range(len(self.vecs_list)),self.r)
        
        return list(combs)
    
    
    
    def get_depths(self):
        
        '''Calculates the depth values for each of the member of the ensemble
        in self.vecs_list
        
        Inputs:
            None
        
        Returns:
            depths- list of depth values corresponding to self.vecs_list
        
        '''
        
        
        ensize = len(self.vecs_list)
        dprint('ensize',ensize)
        depths = np.zeros(ensize)
        combs = self.get_combinations()
        vec_len = len(self.vecs_list[0])


        for i in range(ensize):
            inband_count = 0

            for comb in combs:

                subset = self.vecs_list[comb,:]
                  
                cur_vec = self.vecs_list[i,:]
                max_vec = np.amax(subset, axis = 0)
                min_vec = np.amin(subset, axis = 0)
                top_bounded = cur_vec<=max_vec
                bot_bounded = cur_vec>=min_vec
                  
                inside = top_bounded & bot_bounded
                
                if self.epsilon == 1:
                    if (np.sum(inside) == vec_len):
                        inband_count = inband_count + 1
                elif self.epsilon == -1:
                        inband_count = inband_count + float(np.sum(inside))/vec_len
                else:
                    inside_frac = np.sum(inside)/float(vec_len)
                    if inside_frac >= self.epsilon:
                        inband_count+=1

            depths[i] = inband_count/float(len(combs))

            
        return depths


