'''
Created on 2016-08-10

Functions related to reading the data converted to csv files.

@author: mraj

'''

from produtils import dprint
from src.settings import PROJ_ROOT
import numpy as np
import csv
from pylab import *
from sklearn.metrics import euclidean_distances
import networkx as nx
import json
from sklearn import preprocessing

def get_weather_ensemble(datapath, startyear, endyear, colnum,divby=1):
    """

    Args:
        startyear: start year
        endyear: end year
        colnum: The column to pick from the text files.
        divby: To pick every divby-th value.

    Returns:


    """
    # todo: fix the issue of leap year

    functions = []

    for yr in range(startyear,endyear):
        features = np.genfromtxt(PROJ_ROOT+datapath+str(yr)+'.csv', delimiter=',')
        # features = np.transpose(features)
        # dprint(np.shape(features[:,colnum]))

        functions.append(np.squeeze(features[0:365:divby,colnum]))

    arr= np.array(functions)
    dprint(arr.shape)
    # dprint(arr)

    return functions


def get_microarray_ensemble(features_file,class_file, N_class):
    """Reader function to read hd microarray and return in a function ensemble
    form.

    Args:
        features_file (str): Path (with filename) to the features csv file.
        class_file (str): Path (with filename) to the class file.
        N_class (dict): A dict with class name as key and requested number
          from that class as the value.

    Returns:


    """
    functions = []

    features = np.genfromtxt(PROJ_ROOT+features_file, delimiter=',')

    with open(PROJ_ROOT+class_file, 'rb') as f:
        reader = csv.reader(f)
        classes = list(reader)
    classes = [item for sublist in classes for item in sublist]

    for key, value in N_class.iteritems():

        i = 0
        ctr = 0
        while(ctr<value or i == len(classes)):
            i = i+1
            if classes[i] == key:
                functions.append(features[i,1:500])
                ctr = ctr + 1


    return functions


def get_mutag_gram_matrix(csv_file,N1,N2):
    """

    Args:
        csv_filepath: full path and filename
        N1: Members to pick from class 1 . Must be 0<=125
        N2: Members to pick from class 1 . Must be 0<=63

    Returns:
        G: gram matrix

    """

    inds = range(N1)
    inds.extend(range(125,125+N2))
    inds = np.array(inds)
    G = np.loadtxt(csv_file, delimiter=',')
    dprint(np.shape(G))
    G = G[:,inds]
    G = G[inds,:]

    return G


import matplotlib.pyplot as plt
def get_mnist_data_from_csv(digit,N=150):
    """

    Args:
        digit: The number to fetch.

    Returns:

    """

    filepath = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/mnist/train.csv"

    data = []
    digits = np.zeros((N,28,28))

    with open(filepath, 'rb') as f:
        reader = csv.reader(f)

        next(reader)
        cur_N = 0
        for row in reader:

            row = map(int, row)
            label = row[0]
            if label==digit:
                mat = np.array(row[1:])

                mat = mat.reshape((28,28),order='C')
                mat = 255-mat
                digits[cur_N::] = mat

                data.append(row[1:])
                cur_N += 1

            if cur_N==N:
                break

        data = np.array(data)



    return data,digits

def get_npz_mnist_data_from_csv():
    """

    Returns:

    """

    filepath = "/Users/mukundraj/Desktop/work/projects/kerneldepth/kerneldepth/data/2017-01-23/mnist/train.csv"

    data = []
    # digits = np.zeros((N,28,28))
    counts = np.zeros(10)

    with open(filepath, 'rb') as f:
        reader = csv.reader(f)

        next(reader)

        cur_set = 0
        cur_N = 0
        for row in reader:

            row = map(int, row)

            data.append(row)
            # label = row[0]
            # counts[label] +=1

            # if label==digit:

                # mat = np.array(row[1:])
                #
                # mat = mat.reshape((28,28),order='C')
                # mat = 255-mat
                # digits[cur_N::] = mat
                #
                # data.append(row[1:])
                # cur_N += 1

            # if cur_N==N:
            #     break

        # data = np.array(data)
        data = np.array(data)
        np.savez('mnist_higheres.npz', data)

        dprint(np.shape(data))

    return counts

def get_mnist_data_from_npz(digit, setsize, setid):
    """Get digit data.

    Args:
        digit:
        setsize:
        setid:

    Returns:

    """
    N = setsize
    npzpath = "../../../../data/2017-01-23/mnist/mnist_higheres.npz"
    npzfile = np.load(npzpath)

    data = npzfile['arr_0']



    digit_inds = data[:,0]==digit
    digit_data = data[digit_inds,1:]



    dprint(np.shape(digit_data))

    startid = N*setid

    digit_data = digit_data[startid:startid+N,:]
    digits = np.zeros((N,28,28))
    for i in range(N):
        mat = digit_data[i,:]
        mat = mat.reshape((28,28),order='C')
        mat = 255-mat
        digits[i::] = mat


    return digit_data,digits


def write_tensorflow_csv(points):
    """ A writer for loading data into http://projector.tensorflow.org/

    Args:
        points: A list of d dimensional points.

    Returns:

    """

    points = np.array(points)
    np.savetxt('data.tsv', points, delimiter='\t')


def write_distances_tsv(points, filename, points_ori = None):
    """ Writes the distances between points into a tsv

    Args:
        points_mds:

    Returns:

    """

    similarities = euclidean_distances(points, points)

    if points_ori is not None:
        sim_ori = euclidean_distances(points_ori)
        # sim_ori = sim_ori/np.amax(sim_ori)

    m,n = np.shape(similarities)


    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = ['v0', 'v1', 'dis', 'disori']
        writer.writerow(header)
        # for i in range(N):
        #     row = [round(xs[i],4), round(ys[i],4), round(xs_new[i],4), round(ys_new[i],4),
        #            round(dfgd[i,0],4), round(dfgd[i,1],4), round(dgd[i,0],4), round(dgd[i,1],4),
        #            round(self.depths[i],4)]
        #     writer.writerow(row)

        for i in range(m):
            for j in range(i):
                row = [str(i), str(j), round(similarities[i,j],3)]
                if points_ori is not None:
                    row.append(round(sim_ori[i,j],3))
                writer.writerow(row)


def write_edges_tsv(edges, filename):
    """Writes the edge information to a tsv file

    Args:
        edges: list of edges

    Returns:

    """

    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = ['v0', 'v1']
        writer.writerow(header)


        for edge in edges:
            row = [edge[0], edge[1]]
            writer.writerow(row)

def read_london_tube_tsv(path):
    """

    Args:
        path: The path to the csv files

    Returns:
        G: A networkx graph with weights
    """
    fullpath = path+"london.connections.csv"
    edge_data = np.genfromtxt(fullpath, delimiter=',', skip_header=1)

    m,n = np.shape(edge_data)
    G = nx.Graph()

    for i in range(m):
        G.add_edge(edge_data[i,0],edge_data[i,1],weight=edge_data[i,3])


    # paths=nx.all_pairs_dijkstra_path_length(G, weight='weight')


    return G


def write_node_names_to_json(node_names, array_name='node_names', node_values=None):
    """Cleans up and writes the node names into a json file of name: node_names.json

    Args:
        node_names: A list of node names
        array_name: if we need to write classification info instead of node_names (added later)
        node_values: if we need to also write a node attribute. eg. as in case of polbooks
    Returns:

    """
    node_names = map(str, node_names)
    node_names = map(str.rstrip, node_names)
    node_names = map(lambda s:  ''.join(e for e in s if e.isalnum() or e.isspace()), node_names)
    output_dict = {}
    output_dict[array_name] = node_names
    if node_values is not None:
        output_dict['node_values'] = node_values
    with open(array_name+'.json', 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)

def read_depth_and_pos_from_tsv(tsvfile):
    """Reads and returns depth and position data in the appropriate format.

    Args:
        tsvfile:

    Returns:

    """

    data = np.loadtxt(tsvfile,skiprows=1,usecols=(5,10,11))

    data = np.array(data)

    depths = data[:,0]
    pos = []

    for i in range(len(depths)):
        pos.append(data[i,1:])

    return depths,pos

def read_pos_from_tsv(tsvpath):
    """For reading from the iterations tsvs. Keeping in mind to read the tsv
    with MDS positions.

    Args:
        tsvpath:

    Returns:

    """

    data = np.loadtxt(tsvpath,skiprows=1,usecols=(0,1))

    data = np.array(data)

    pos = []

    for i in range(len(data)):
        pos.append(data[i,:])

    return pos

def get_ufo_data(infile):
    """

    Args:
        infile: path to csv file

    Returns:
        X: 2D numpy array

    """

    le = preprocessing.LabelEncoder()

    X, y = [], []

    with open(infile) as f:
        lines = f.read().splitlines()
        for line in lines:
            seq = line.decode('ascii').split('\t')
            X.append([x.strip() for x in seq[2:]])

    countries = [x[0] for x in X]
    le.fit(countries)
    countries = le.transform(countries)

    for i in range(len(X)):
        X[i][0] = chr(countries[i]+97)
        X[i] = [ord(x.strip()) for x in X[i]]

    X = np.array(X)
    dprint(np.shape(X))
    return X

def get_breast_data(infile, clas=0):
    """
    Args:
        infile: Path to csv file.

    Returns:
        X: 2D numpy array
        clas: 0=all, 1=no-recurrance-events, 2=recurrence-events

    """

    X, y = [], []
    le = preprocessing.LabelEncoder()

    with open(infile) as f:
        lines = f.read().splitlines()
        for line in lines:
            seq = line.decode('ascii').split(',')
            X.append([x.strip() for x in seq])

    m = len(X)
    n = len(X[0])


    for j in range(n):
        col = [x[j] for x in X]
        le.fit(col)
        col = le.transform(col)

        for i in range(m):
            X[i][j] = chr(col[i]+97)

    for i in range(m):
        X[i] = [ord(x.strip()) for x in X[i]]

    X = np.array(X)

    if clas==1:
        X=X[np.where(X[:,0]==97)]
    elif clas==2:
        X=X[np.where(X[:,0]==98)]

    return X


def write_csv_for_breast_vis(infile, clas):
    """ takes infile and writes rows with field header in each row,
    and places it in the home folder for being accessed by the local
    vis file.

    Args:
        infile:

    Returns:

    """

    X = []
    with open(infile) as f:
        lines = f.read().splitlines()
        for line in lines:
            seq = line.decode('ascii').split(',')
            X.append([x.strip() for x in seq])

    with open('output_tsvs/set_vectors.txt', 'w') as f:

        for i in range(len(X)):
            f.write('age:'+ str(X[i][1])+
                    ',meno:'+ str(X[i][2])+
                    ',tsize:'+ str(X[i][3])+
                    ',invnod:'+ str(X[i][4])+
                    ',invcap:'+ str(X[i][5])+
                    ',deg:'+ str(X[i][6])+
                    ',side:'+ str(X[i][7])+
                    ',quad:'+ str(X[i][8])+
                    ',irrad:'+ str(X[i][9])+'\n')

    X = get_breast_data(infile, clas=clas)

    m,n = np.shape(X)


    with open('output_tsvs/set_codes.txt', 'wb') as csvfile_labels:
            writer_labels = csv.writer(csvfile_labels, delimiter='\t')
            # writer_labels.writerow(["label","index"])
            for i in range(m):
                row = [format(i, '03')+'x']
                row2 = [str(j)+'x' for j in X[i,1:]]
                row.extend(row2)
                writer_labels.writerow(row)


def read_breast_write_json(infile,clas):
    """

    Args:
        infile:

    Returns:

    """
    X = []
    with open(infile) as f:
        lines = f.read().splitlines()
        for line in lines:
            seq = line.decode('ascii').split(',')
            if clas==1:
                if seq[0]=='no-recurrence-events':
                    X.append([x.strip() for x in seq])
            elif clas==2:
                if seq[0]=='recurrence-events':
                    X.append([x.strip() for x in seq])
            else:
                X.append([x.strip() for x in seq])

    output_dict = {}
    output_dict['node_vectors'] = X
    with open('output_tsvs/node_vectors.json', 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def read_ufo_write_json(infile):
    """reads an ip file and writes a processed json for visualization.

    Args:
        ip_path:

    Returns:

    """
    X = []
    with open(infile) as f:
        lines = f.read().splitlines()
        for line in lines:
            seq = line.decode('ascii').split('\t')
            X.append([x.strip() for x in seq[1:]])

    dprint(len(X))
    output_dict = {}
    output_dict['node_vectors'] = X
    with open('output_tsvs/node_vectors.json', 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)


def read_fbfilename_write_json(ip_path):
    """

    Args:
        ip_path:

    Returns:

    """

    f = open(ip_path)
    names = f.readlines()
    f.close()

    names = [l.strip(".mat \n") for l in names]
    node_names = []
    whitelist = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    for i, name in enumerate(names):
        name = ''.join(filter(whitelist.__contains__, name))
        node_names.append(name)

    node_names = map(str, node_names)
    node_names = map(str.rstrip, node_names)
    node_names = map(lambda s:  ''.join(e for e in s if e.isalnum() or e.isspace()), node_names)
    output_dict = {}
    output_dict['node_names'] = node_names

    with open('output_tsvs/node_names.json', 'w') as outfile:
        json.dump(output_dict, outfile, sort_keys=True, indent=4)



def write_pos_and_depths_tsv(pos, depth_hs=None, depths_ellp=None, depth_rect=None):
    """Writes a tvs file with positions and depth values

    Args:
        pos:
        depth_hs:
        depths_ellp:
        depth_rect:

    Returns:

    """
    N = len(pos)
    with open('output_tsvs/data.tsv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = ['cpx', 'cpy', 'dhs', 'dlp']
        writer.writerow(header)
        for i in range(N):
            row = [round(pos[i][0],4), round(pos[i][1],4)]
            if depth_hs is not None:
                row.append(round(depth_hs[i], 4))
            else:
                row.extend(0)
            if depths_ellp is not None:
                row.append(round(depths_ellp[i], 4))
            else:
                row.extend(0)
            writer.writerow(row)


