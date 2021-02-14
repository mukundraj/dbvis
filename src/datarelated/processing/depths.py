'''
Created on Aug 27, 2015

Functions to process depth data.

@author: mukundraj
'''
import copy
import math
import datetime
import time

import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KernelDensity

from libs.productivity import *



def get_median_and_bands(depths, alpha = 1.5):
    """Takes in the depth values and returns the indices of objects
    in various regions in set form.
    
    args:
        depths: An array of depth values.
        alpha: The factor alpha for identifying outliers. If a neative
        integer, then hard code the number of outliers to abs(alpha).
    returns:
        median: Index of median.
        band50: Set of indices in the 50 percent band.
        band100: Set of indices in the 100 percent band.
        outliers: Set of indices for outliers.
        cat_list: A list of categories based on the indices of the depth 
            input parameter.
        band_prob_bounds: A list of upper and lower bounds for each of the
            four band categories. upper- away from zero, lower- near zero
    

    """
    # TODO (mukund): Handle case of the multiple medians.

    sorted_inds = sorted(range(depths.shape[0]), key=lambda k: depths[k])

    median = sorted_inds[-1]
    
    # Get the 50 percent band
    mid = int(math.ceil(len(depths)/2))
    mid_index = sorted_inds[mid]
    band50 = set(sorted_inds[mid:])
    band50.update([median])
    
    # Get the 100 percent band and outliers
    band100 = set()
    outliers = set()
    min_band50_prob = depths[mid_index]
    
    if(alpha<0):
        band100_prob = depths[median] - 1.5*(depths[median] - depths[mid_index])

    else:    
        band100_prob = depths[median] - alpha*(depths[median] - depths[mid_index])

    # dprint('band100_prob', band100_prob)
    for i in sorted_inds:
            depth = depths[i]

            if depth < band100_prob:
                    outliers.add(i)
            elif depth < min_band50_prob: # Handling case with ties
                    band100.add(i)
            elif depth <depths[median]:
                    band50.add(i)

    # dprint(outliers)
    # Fill in required number of outliers for vis testing.
    if alpha < 0:
        i=0
        while(len(outliers)<abs(alpha)):
            outliers.add(sorted_inds[i])
            i = i+1


    band100.update(band50)
    band100.update([median])

    cat_list = []
    for i in range(len(depths)):
        if i in outliers:
            cat_list.append(3)
        elif i in [median]:
            cat_list.append(0)
        elif i in band50:
            cat_list.append(1)
        elif i in band100:
            cat_list.append(2)
        
    
    band_prob_bounds_list = [{"lower":0, "higher":round(float(band100_prob),3)}, #outliers (lower, upper)
                        {"lower":round(float(band100_prob),3), "higher":round(float(min_band50_prob),3)}, #band100 (lower, upper)
                        {"lower":round(float(min_band50_prob),3), "higher":round(float(depths[median]),3)}, #band50 (lower, upper)
                        {"lower":round(float(depths[median]),3),"higher":1}] #median (lower, upper)
    
#     dprint(depths)
#     dprint(sorted_inds)
#     dprint(band_prob_bounds_list)
#     dprint(median,band50,band100,outliers)
    return median, band50, band100, outliers, cat_list, band_prob_bounds_list
    
    
def process_analysis_results(depths, op_file_path = False,
                              xtra_cols = None, xtra_col_labels = None, params = {}):
    """Sorts results based on the depth.
    
    And displays alongside respective indices. Prints if needed. 
    Conversion to pandas data frame. More functionality can be added.
    
    Args:
        depths: An array of depth values. Order as in the original order in
            the original order in ensemble.
        write_to_file_path: Flag to indicate if results are to be written to
             file. To write, path includes filename.
        xtra_cols: List of extra columns to add to output apart from index and depth.
        xtra_col_labels: List of labels for the extra columns.
        params: A dict of parameters
        
    Returns: 
        output: pandas dataframe
        
    """
    # TODO (mukund): Deal with issue of outlier in middle of list due to ties in
    #    depth values.
    # TODO (mukund): Add num of member in bands and threshold cut off explicitly
    # to output file.

    median, band50, band100, outliers, cat_list, band_prob_bounds = \
                                get_median_and_bands(depths, params['alpha'])
    # dprint(median,band50,band)
    ensemble_size = len(depths)
    
    output = np.append(np.array(range(1,ensemble_size+1)).reshape(ensemble_size,1),depths.reshape(ensemble_size,1),1) 
    output = output[output[:,1].argsort()[::-1]]
    columns = ['index', 'depth']
    df = pd.DataFrame(data = output, columns = columns)
    
    if xtra_cols != None:
        for i,column in enumerate(xtra_cols):
            df.loc[:, xtra_col_labels[i]] = xtra_cols[i]    
        
    df[['index']] -= 1 # For index to start from zero
    

    #dprint('\n'.join(map(str, band_prob_bounds)))
    file_handle = open(op_file_path, 'w')
    
    # Write output
    
    if op_file_path != False:
        df['depth'] = df['depth'].map(lambda x: '%.3f' % x)
        df.to_csv(file_handle, sep='\t', index = False)
    
    file_handle.write('## Parameters ##\n')    
    for param,value in params.iteritems():
        if param!='elapsed':
            file_handle.write('# ' + param + ':\t' + str(value) + '\n')
    file_handle.write('## Other Info ##\n')
    ts = time.time()  
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    file_handle.write('# Timestamp' + ':\t' + st + '\n')
    file_handle.write('# Analysis runtime' + ':\t' + str(params['elapsed'])
                      + ' seconds\n')
    file_handle.write('# median' + ':\t' + str(median)  + '\n')
    file_handle.write('# band50' + ':\t' + str(band50)  + '\n')
    file_handle.write('# band100' + ':\t' + str(band100)  + '\n')
    file_handle.write('# outliers' + ':\t' + str(outliers)  + '\n')
    file_handle.write('# bounds' + ':\t' + '\n#\t\t\t'.join(map(str, reversed(band_prob_bounds)))+ '\n')

    file_handle.close()


    
    return df

def get_dists_from_median(points, depths):
    """Gets an array of dists from median

    Args:
        points:
        depths:

    Returns:
        dists: Array of dists from the median

    """
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(np.array(depths))

    all_dists = euclidean_distances(points, points)

    dists = all_dists[median,:]

    return dists

def get_dists_from_median_graph_version(distmat, depths):
    """Gets an array of dists from median. This version is for graph distances.

    Args:
        distmat:
        depths:

    Returns:

    """
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(np.array(depths))

    dists = distmat[median,:]

    return dists

def histeq(arr, number_bins=20, bw=0.02, res=1000):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # # get image histogram
    # image_histogram, bins = np.histogram(arr, number_bins, normed=True)
    # cdf = image_histogram.cumsum() # cumulative distribution function
    # cdf = 1 * cdf / cdf[-1] # normalize
    #
    # # use linear interpolation of cdf to find new pixel values
    # image_equalized = np.interp(arr, bins[:-1], cdf)

    xs = np.linspace(0,1,res)
    n = len(arr)

    a1 = copy.deepcopy(arr).reshape((n,1))
    kde_a1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a1)
    a1s = np.exp(kde_a1.score_samples(xs.reshape((res,1))))
    a1s = a1s/np.sum(a1s) # normalize for the grid size

    cdf = a1s.cumsum()
    cdf = 1 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    dep_eq = np.interp(arr, xs, cdf)

    image_equalized = dep_eq

    return image_equalized