from networks import get_network, get_average_degree
from dla_rand import DLA_init, plot_grid
import networkx as nx
import numpy as np
import pandas as pd
import os
import seaborn as sns

directory = "results/changing_stickness/"

# read the files, convert it to networks and get statistics

files = os.listdir(directory)
df = pd.DataFrame()

for file in files:
    try:
        grid = np.load(directory + file, allow_pickle = True)
    except OSError:
        continue
    # get simulation number and stickness
    stickiness = float(file[0:file.find(",")])
    simulation = (file[file.find(",")+1:file.find(".")])
    # convert grid to a network
    G = get_network(grid, simplify = True)
    average_degree = get_average_degree(G)
    average_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    diameter = nx.diameter(G)
    radius = nx.radius(G)
    # create a df dictionary
    df_new = pd.DataFrame({
        "stickiness" : [stickiness],
        "simulation" : [simulation],
        "average_degree" : [average_degree],
        "average_clustering" : [average_clustering],
        "transitivity" : [transitivity],
        "diameter": [diameter],
        "radius" : [radius]
        })
    df = df.append(df_new)
    

    
    