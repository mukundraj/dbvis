'''
Created on 2016-05-07

Functions for generating tsvs an jsons for web based visualizations.

@author: mraj

'''
from libs.productivity import dprint
import csv
from itertools import combinations
import numpy as np
import src.analysis.bands as bands
from datarelated.processing.depths import get_median_and_bands
import json
from scipy.spatial import ConvexHull
# import libs.smallestenclosingcircle as smcir
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from sklearn.decomposition import PCA
import src.testcode.smallestenclosingcircle as smcir

def write_json_tsv_for_boxplot(depths_dict, vectors, tsv_filename, json_filename, alpha, depths_grid_dict):
    """ Writes a tsv and a json file.

    Args:
        vectors: An ensemble of vectors.

    Returns:
        None

    """

    num_dims = len(vectors[0])
    num_members = len(vectors)

    depths_rect = depths_dict["depths_rect"]
    median, band50, band100, outliers, cat_list_rect, band_prob_bounds_rect = get_median_and_bands(depths_rect, alpha=alpha)

    depths_ellp = depths_dict["depths_ellp"]
    median, band50, band100, outliers, cat_list_ellp, band_prob_bounds_ellp = get_median_and_bands(depths_ellp, alpha=alpha)

    depths_sphere = depths_dict["depths_sphere"]
    median, band50, band100, outliers, cat_list_sphere, band_prob_bounds_sphere = get_median_and_bands(depths_sphere, alpha=alpha)

    depths_hull = depths_dict["depths_hull"]
    median, band50, band100, outliers, cat_list_hull, band_prob_bounds_hull = get_median_and_bands(depths_hull, alpha=alpha)


    with open(tsv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = []
        for i in range(num_dims):
            header.append('g' + str(i))
        header.extend(["depths_rect", "depths_ellipse", "depths_circle", "depths_hull"])
        header.extend(["cat_rect", "cat_ellipse", "cat_sphere", "cat_hull"])
        writer.writerow(header)

        for i in range(num_members):
            row = []
            row.extend([round(elem,3) for elem in vectors[i]])
            row.extend([round(x,3) for x in [depths_rect[i],depths_ellp[i],depths_sphere[i],depths_hull[i]]])
            row.extend([cat_list_rect[i], cat_list_ellp[i], cat_list_sphere[i], cat_list_hull[i]])
            writer.writerow(row)

    # Depth levels for rectangle
    zs_rect = [0.0]
    for bound in band_prob_bounds_rect:
        zs_rect.append(round(max(0,bound["higher"]-0.02),3))
    zs_rect[-1]=1
    depths_grid_dict["zs_rect"] = zs_rect

    # Depth levels for ellipse
    zs_ellp = [0.0]
    for bound in band_prob_bounds_ellp:
        zs_ellp.append(round(max(0,bound["higher"]-0.02),3))
    zs_ellp[-1]=1
    depths_grid_dict["zs_ellp"] = zs_ellp

    # Depth levels for sphere
    zs_sphere = [0.0]
    for bound in band_prob_bounds_sphere:
        zs_sphere.append(round(max(0,bound["higher"]-0.02),3))
    zs_sphere[-1]=1
    depths_grid_dict["zs_sphere"] = zs_sphere

    # Depth levels for hull
    zs_hull = [0.0]
    for bound in band_prob_bounds_hull:
        zs_hull.append(round(max(0,bound["higher"]-0.02),3))
    zs_hull[-1]=1
    depths_grid_dict["zs_hull"] = zs_hull


    with open(json_filename, 'w') as outfile:
        json.dump(depths_grid_dict, outfile, sort_keys=True, indent=4)


def write_tsv_for_band(vectors, tsv_filename, r):
    """Writes the tsv file with band info. For example, if the bands are ellipses
    then for each band writes the parameters needed to draw the ellipse

    Args:
        vectors:
        tsv_filename:
        r: Number of members forming the band.

    Returns:
        None

    """

    shape = np.shape(vectors)
    combs = list(combinations(range(shape[0]),r))

    vectors_nparray = np.array(vectors)

    dprint(len(combs), combs)

    with open(tsv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = []
        for i in range(r):
            header.append('m' + str(i))

        header.extend(['xmin','xmax','ymin','ymax'])
        header.extend(['cx','cy','ra','rb','rot'])
        header.extend(['ccx','ccy','r'])
        header.extend(['simplices'])

        # for i in range(r):
        #     header.append('x' + str(i))
        #     header.append('y' + str(i))
        writer.writerow(header)



        for i in range(len(combs)):
            xs = [round(vectors[memberid][0],3) for memberid in combs[i]]
            ys = [round(vectors[memberid][1],3) for memberid in combs[i]]
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
            row = []
            row.extend([memberid for memberid in combs[i]])
            row.extend([xmin, xmax, ymin, ymax])


            # Now ellipse band
            ell_info = bands.get_ellipse2d_info(vectors_nparray[combs[i],:])
            row.extend([ell_info['cx'],ell_info['cy'],ell_info['ra'],ell_info['rb'],ell_info['rot']])

            # Now circle band
            C = smcir.make_circle(vectors_nparray[combs[i],:])
            row.extend([round(x,3) for x in C])

            # Now hull
            cur_pts = vectors_nparray[combs[i],:]
            hull = MultiPoint(cur_pts).convex_hull
            xs, ys = hull.exterior.coords.xy
            xs = [round(x,3) for x in xs]
            ys = [round(y,3) for y in ys]
            sim = [xs,ys]
            # hull = ConvexHull(cur_pts)
            # sims = [[list(cur_pts[i][1]),list(cur_pts[j])] for i,j in hull.simplices]
            # sim = [list(x) for x in sims]
            row.extend([sim])

            #dprint(ell_info)
            # pairs = [[round(vectors[memberid][0],3),round(vectors[memberid][1],3)] for memberid in combs[i]]
            # for pair in pairs:
            #     row.extend(pair)
            #
            writer.writerow(row)


def write_tsv_funcspaghetti(functions, outfilename):
    """Writes a tsv for d3 driven functional spaghetti plot.

    Also handles conversion from graphs to functions.

    Args:
        functions (list): a list of functions(1D numpy array).
        outfilename (string): Name of output file. Add .json to end. Add path if needed.

    """

    ensize = len(functions)
    domainsize = len(functions[0])


    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = ['x']
        for i in range(ensize):
            header.append('y' + str(i))
        writer.writerow(header)

        for dom in range(domainsize):
            row = [dom]
            for mid in range(ensize):
                row.extend([round(functions[mid][dom],3)])
            writer.writerow(row)


def write_json_funcbandinfo(functions, depths_dict, json_filename, alpha, rval, fulllist=None):
    """Writes json specifying the functions the
     50,100 percent bands. Furthermore, stores the maxval, minval
    and ensize for retrieval in the d3 based visualization.

    Args:
        depths_dict:
        json_filename:
        fulllist: All the members + the grid functions.

    Returns:

    """
    output_dict = {}


    functions_nparray = np.array(functions)
    max_val = np.amax(functions_nparray)
    min_val = np.amin(functions_nparray)
    totsize = len(depths_dict["depths_rect"])

    output_dict["max_val"] = round(max_val,3)
    output_dict["min_val"] = round(min_val,3)
    output_dict["ensize"] = len(functions)
    output_dict["rval"] = rval
    output_dict["totsize"] = totsize

    ensize = len(functions)

    # Get band indices for rect band
    depths_rect = depths_dict["depths_rect"]
    median, band50, band100, outliers, cat_list_rect, band_prob_bounds_rect = get_median_and_bands(depths_rect[:ensize], alpha=alpha)
    output_dict["rect_median"] = [median]
    output_dict["rect_band50"] = list(band50)
    output_dict["rect_band100"] = list(band100)
    output_dict["rect_outliers"] = list(outliers)

    depths_ellp = depths_dict["depths_ellp"]
    median, band50, band100, outliers, cat_list_ellp, band_prob_bounds_ellp = get_median_and_bands(depths_ellp[:ensize], alpha=alpha)
    output_dict["ellp_median"] = [median]
    output_dict["ellp_band50"] = list(band50)
    output_dict["ellp_band100"] = list(band100)
    output_dict["ellp_outliers"] = list(outliers)

    pca_list = []
    if fulllist:
        pca = PCA(n_components=2)
        dprint(np.shape(fulllist))
        pca_pos = pca.fit_transform(fulllist)
        dprint(np.shape(pca_pos))
        output_dict["pca_max"] = np.amax(pca_pos)
        output_dict["pca_min"] = np.amin(pca_pos)

        dprint(len(pca_pos))
        for i in range(totsize):
            pca_list.append({"id":i, "pos": list(pca_pos[i,:])})
            if i < ensize:
                pca_list[i]["cat_rect"] = cat_list_rect[i]
                pca_list[i]["cat_ellp"] = cat_list_ellp[i]
            else:
                pca_list[i]["cat_rect"] = -1
                pca_list[i]["cat_ellp"] = -1


    output_dict["pca_pos"] = pca_list

    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def write_json_funcbandinfo_kernelversion(functions, Kxx_proj, depths_dict, json_filename, alpha, rval, fulllist=None):
    """Writes json specifying the functions the
     50,100 percent bands. Furthermore, stores the maxval, minval
    and ensize for retrieval in the d3 based visualization.

    Kernel version takes into account the extra "grid" function depths and ignores them.

    Args:
        depths_dict:
        json_filename:
        fulllist: actually same as Kxx_proj. Redundancy to maintain naming consistancy with nonkernel
            version of this function.

    Returns:

    """
    output_dict = {}

    functions_nparray = np.array(functions)
    max_val = np.amax(functions_nparray)
    min_val = np.amin(functions_nparray)

    totsize = len(Kxx_proj)
    ensize = len(functions)

    output_dict["max_val"] = round(max_val,3)
    output_dict["min_val"] = round(min_val,3)
    output_dict["ensize"] = len(functions)
    output_dict["totsize"] = len(Kxx_proj)
    output_dict["rval"] = rval

    # Get band indices for rect band
    depths_rect = depths_dict["depths_rect"][:len(functions)]
    median, band50, band100, outliers, cat_list_rect, band_prob_bounds_rect = get_median_and_bands(depths_rect, alpha=alpha)
    output_dict["rect_median"] = [median]
    output_dict["rect_band50"] = list(band50)
    output_dict["rect_band100"] = list(band100)
    output_dict["rect_outliers"] = list(outliers)


    # Get band indices for k1 band
    depths_k2 = depths_dict["depths_k2"][:len(functions)]
    median, band50, band100, outliers, cat_list_k1, band_prob_bounds_rect = get_median_and_bands(depths_k2, alpha=alpha)
    output_dict["k2_median"] = [median]
    output_dict["k2_band50"] = list(band50)
    output_dict["k2_band100"] = list(band100)
    output_dict["k2_outliers"] = list(outliers)


    pca_list = []
    if fulllist:
        pca = PCA(n_components=2)
        dprint(np.shape(fulllist))
        pca_pos = pca.fit_transform(fulllist)
        dprint(np.shape(pca_pos))
        output_dict["pca_max"] = np.amax(pca_pos)
        output_dict["pca_min"] = np.amin(pca_pos)

        dprint(len(pca_pos))
        for i in range(totsize):
            pca_list.append({"id":i, "pos": list(pca_pos[i,:])})
            if i < ensize:
                pca_list[i]["cat_rect"] = cat_list_rect[i]
                pca_list[i]["cat_ellp"] = cat_list_k1[i]
            else:
                pca_list[i]["cat_rect"] = -1
                pca_list[i]["cat_ellp"] = -1


    output_dict["pca_pos"] = pca_list


    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def write_tsv_funcband(bands_dict, tsv_band_filename, r):
    """

    Args:
        bands_dict:
        tsv_band_filename:
        r:

    Returns:

    """

    # ensize = len(functions)
    # dprint(ensize)
    # combs = list(combinations(range(ensize),r))
    #
    #
    #
    # dprint(len(combs), combs)


    with open(tsv_band_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = []
        header.extend(['index', 'comb', 'rect', 'ellp'])

        writer.writerow(header)

        for i, key in enumerate(bands_dict["bands_rect_j"]):
            row = [i, key, bands_dict["bands_rect_j"][key], bands_dict["bands_ellp"][key]]
            writer.writerow(row)

        # for i in range(len(combs)):
        #     row = [str(list(combs[i]))]
        #     row.append(str([1,2]))
        #     row.append(str([4,5,6]))



def write_tsv_kernel_band(inside_list, inside_list_rect, tsv_band_filename, combs, points2d = None):
    """

    Args:
        tsv_band_filename:

    Returns:

    """

    if points2d is not None:
        points2d = np.array(points2d)
    if inside_list_rect is None:
        inside_list_rect = inside_list

    with open(tsv_band_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = []
        header.extend(['index', 'comb', 'band', 'band_rect'])

        if points2d is not None:
            header.extend(['cx','cy','ra','rb','rot'])

        writer.writerow(header)

        for i in range(len(inside_list)):
            row = [i, list(combs[i]), list(inside_list[i]), list(inside_list_rect[i])]

            # Only for the points case
            if points2d is not None:
                # Now ellipse band
                ell_info = bands.get_ellipse2d_info(points2d[combs[i],:])
                row.extend([ell_info['cx'],ell_info['cy'],ell_info['ra'],ell_info['rb'],ell_info['rot']])

            writer.writerow(row)

def write_tsv_kernel_enumeration(depths, alpha, Kxx_proj, outfilename, points_list):
    """Writes a tsv for d3 driven plotting of points with kernel band/depth.

    Args:
        Kxx_proj: The combined projections.
        outfilename: The output file name.

    """


    N = len(points_list)
    num_members = len(Kxx_proj)
    num_dims = len(Kxx_proj[0])


    median, band50, band100, outliers, cat_list, band_prob_bounds = get_median_and_bands(depths[:N], alpha=alpha)
    # dprint(N,cat_list, median,band50,band100,outliers)
    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = []
        for i in range(num_dims):
            header.append('g' + str(i))
        header.extend(['isgrid', 'k1cat','depths_hs'])
        writer.writerow(header)

        # for dim in range(domainsize):
        #     row = [dim]
        #     for fid in range(ensize):
        #         row.extend([round(Kxx_proj[fid][dim],3)])
        #     writer.writerow(row)

        for i in range(num_members):
            row = []
            isgrid = 1
            k1_cat = -1
            if i<N:
                isgrid=0
                k1_cat=cat_list[i]
            row.extend([round(Kxx_proj[i][0],3), round(Kxx_proj[i][1],3), isgrid, k1_cat,depths[i]])
            writer.writerow(row)

def write_json_kernel_point_bandinfo(json_filename, pts_list, Kxx_proj, rval):
    """Write json with auxillary information for kernel point visualization.

    Args:


    Returns:

    """
    output_dict = {}

    N = len(pts_list)
    N_total = len(Kxx_proj)


    output_dict["ensize"] = N
    output_dict["totsize"] = N_total
    output_dict["rval"] = rval

    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def write_json_halfspace_point_bandinfo(json_filename, pts_list, Kxx_proj, N_directions,
                                        depths_dict=None, alpha=1.5, N_i=None):
    """Write json with auxillary information for kernel point visualization. This is the
    half space depth version and does not include the band size 'r'.

    Args:
        N_directions: Number of directions.

    Returns:

    """
    output_dict = {}

    N = len(pts_list)
    N_total = len(Kxx_proj)


    output_dict["ensize"] = N
    output_dict["totsize"] = N_total
    output_dict["N_directions"] = N_directions

    # Store band indices for pre and post mds analysis bands
    depths_pre = depths_dict["khs_pre"][:len(pts_list)]
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths_pre, alpha=alpha)
    output_dict["khs_pre_median"] = [median]
    output_dict["khs_pre_band50"] = list(band50)
    output_dict["khs_pre_band100"] = list(band100)
    output_dict["khs_pre_outliers"] = list(outliers)

    sorted_inds = sorted(range(depths_pre.shape[0]), key=lambda k: depths_pre[k])
    output_dict["khs_pre_sorted_inds"] = list(sorted_inds[::-1])

    depths_post = depths_dict["khs_post"][:len(pts_list)]
    median, band50, band100, outliers, cat_list_post, band_prob_bounds_post = get_median_and_bands(depths_post, alpha=alpha)
    output_dict["khs_post_median"] = [median]
    output_dict["khs_post_band50"] = list(band50)
    output_dict["khs_post_band100"] = list(band100)
    output_dict["khs_post_outliers"] = list(outliers)

    if N_i is not None:
        output_dict["N_i"] = N_i


    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def write_tsv_halfspace(inside_list, tsv_band_filename, inside_list_post=None):
    """

    Args:
        inside_list : inside list default or pre analysis
        inside_list_post: inside list post analysis
        tsv_band_filename:

    Returns:

    """

    with open(tsv_band_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = []
        header.extend(['index', 'halfspace_members', 'halfspace_members_post'])

        writer.writerow(header)

        for i in range(len(inside_list)):
            row = [i,  list(inside_list[i])]
            if inside_list_post is not None:
                row.append(inside_list_post[i])

            writer.writerow(row)


def write_json_funcbandinfo_halfspace(functions, depths_dict, json_filename, alpha):
    """Writes json specifying the functions the
     50,100 percent bands for halfspace depth. Furthermore, stores the maxval, minval
    and ensize for retrieval in the d3 based visualization.

    Kernel version takes into account the extra "grid" function depths and ignores them.

    Args:
        depths_dict:
        json_filename:


    Returns:

    """
    output_dict = {}

    functions_nparray = np.array(functions)
    max_val = np.amax(functions_nparray)
    min_val = np.amin(functions_nparray)

    ensize = len(functions)

    output_dict["max_val"] = round(max_val,3)
    output_dict["min_val"] = round(min_val,3)
    output_dict["ensize"] = len(functions)

    # Store band indices for pre and post mds analysis bands
    depths_pre = depths_dict["khs_pre"][:len(functions)]
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths_pre, alpha=alpha)
    output_dict["khs_pre_median"] = [median]
    output_dict["khs_pre_band50"] = list(band50)
    output_dict["khs_pre_band100"] = list(band100)
    output_dict["khs_pre_outliers"] = list(outliers)


    # Get band indices for k1 band
    depths_post = depths_dict["khs_post"][:len(functions)]
    median, band50, band100, outliers, cat_list_post, band_prob_bounds_post = get_median_and_bands(depths_post, alpha=alpha)
    output_dict["khs_post_median"] = [median]
    output_dict["khs_post_band50"] = list(band50)
    output_dict["khs_post_band100"] = list(band100)
    output_dict["khs_post_outliers"] = list(outliers)


    # pca_list = []
    # if fulllist:
    #     pca = PCA(n_components=2)
    #     dprint(np.shape(fulllist))
    #     pca_pos = pca.fit_transform(fulllist)
    #     dprint(np.shape(pca_pos))
    #     output_dict["pca_max"] = np.amax(pca_pos)
    #     output_dict["pca_min"] = np.amin(pca_pos)
    #
    #     dprint(len(pca_pos))
    #     for i in range(totsize):
    #         pca_list.append({"id":i, "pos": list(pca_pos[i,:])})
    #         if i < ensize:
    #             pca_list[i]["cat_rect"] = cat_list_rect[i]
    #             pca_list[i]["cat_ellp"] = cat_list_k1[i]
    #         else:
    #             pca_list[i]["cat_rect"] = -1
    #             pca_list[i]["cat_ellp"] = -1


    # output_dict["pca_pos"] = pca_list


    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)

def write_tsv_halfspace_highd(depths_dict, alpha, outfilename, points_list,
                              points_list_oc=None, points_list_mds=None):
    """Writes a tsv for d3 driven plotting of boxplots for high dim objects.
    *Note that this takes in as arg depths_dict to handle multiple depths.

    Args:
        depths_dict: Dict of depths.
        outfilename: The output file name.


    """


    N = len(points_list)
    num_members = len(points_list)
    num_dims = len(points_list[0])


    median, band50, band100, outliers, cat_list_pre, band_prob_bounds = get_median_and_bands(depths_dict['khs_pre'], alpha=alpha)
    median, band50, band100, outliers, cat_list_post, band_prob_bounds = get_median_and_bands(depths_dict['khs_post'], alpha=alpha)


    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = ['id']
        for i in range(num_dims):
            header.append('g' + str(i))


        header.extend(['isgrid', 'depth_pre_cat','depths_pre', 'depth_post_cat', 'depths_post'])

        if points_list_oc is not None:
                header.extend(['c0','c1'])
        if points_list_mds is not None:
            header.extend(['m0','m1'])

        writer.writerow(header)

        isgrid=0
        for i in range(num_members):
            row = []
            row.extend([i, round(points_list[i][0],3), round(points_list[i][1],3), isgrid,
                        cat_list_pre[i],round(depths_dict['khs_pre'][i],3), cat_list_post[i],round(depths_dict['khs_post'][i],3)] )
            if points_list_oc is not None:
                row.extend([round(points_list_oc[i][0],3), round(points_list_oc[i][1],3)])
            if points_list_mds is not None:
                row.extend([round(points_list_mds[i][0],3), round(points_list_mds[i][1],3)])

            writer.writerow(row)


def write_tsv_halfspace_abstract(depths_dict, alpha, outfilename, points_list,
                              points_list_oc=None, N_all=None, points_list_all=None,
                                 points_list_all_polar=None):
    """Writes a tsv for d3 driven plotting of boxplots for abstract objects.
    *Note that this takes in as arg depths_dict to handle multiple depths (if needed)

    Args:
        depths_dict: Dict of depths.
        outfilename: The output file name.


    """


    N = len(points_list)
    num_members = len(points_list)
    num_dims = len(points_list[0])



    median, band50, band100, outliers, cat_list_pre, band_prob_bounds = get_median_and_bands(depths_dict['khs_pre'], alpha=alpha)

    if N_all is not None:
        cat_list_pre += [4]*(N_all-N)
        num_members = N_all
        points_list = points_list_all
        depths_dict['khs_pre'] = depths_dict['khs_pre_all']

    dprint(N_all)
    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        header = ['id']
        for i in range(num_dims):
            header.append('g' + str(i))


        header.extend(['isgrid', 'depth_pre_cat','depths_pre'])

        if points_list_oc is not None:
                header.extend(['c0','c1'])

        if points_list_all_polar is not None:
                header.extend(['p0','p1'])

        writer.writerow(header)

        isgrid=0
        for i in range(num_members):
            row = []
            row.extend([i, round(points_list[i][0],3), round(points_list[i][1],3), isgrid,
                        cat_list_pre[i],depths_dict['khs_pre'][i]])
            if points_list_oc is not None:
                row.extend([round(points_list_oc[i][0],3), round(points_list_oc[i][1],3)])
            if points_list_all_polar is not None:
                row.extend([round(points_list_all_polar[i][0],3), round(points_list_all_polar[i][1],3)])

            writer.writerow(row)

def write_json_halfspace_abstract_bandinfo(json_filename, pts_list, N_directions,
                                        depths_dict=None, alpha=1.5):
    """Write json with auxillary information for kernel point visualization. This is the
    half space depth version and does not include the band size 'r'. This version
    is for abstract data.

    Args:
        N_directions: Number of directions.

    Returns:

    """
    output_dict = {}

    N = len(pts_list)


    output_dict["ensize"] = N
    output_dict["totsize"] = N
    output_dict["N_directions"] = N_directions

    # Store band indices for pre and post mds analysis bands
    depths_pre = depths_dict["khs_pre"][:len(pts_list)]
    median, band50, band100, outliers, cat_list_pre, band_prob_bounds_pre = get_median_and_bands(depths_pre, alpha=alpha)
    output_dict["khs_pre_median"] = [median]
    output_dict["khs_pre_band50"] = list(band50)
    output_dict["khs_pre_band100"] = list(band100)
    output_dict["khs_pre_outliers"] = list(outliers)

    sorted_inds = sorted(range(depths_pre.shape[0]), key=lambda k: depths_pre[k])
    output_dict["khs_pre_sorted_inds"] = list(sorted_inds[::-1])

    with open(json_filename, 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)