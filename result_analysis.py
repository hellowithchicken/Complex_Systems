from networks import get_network, get_average_degree, get_entropy, get_dead_ends, get_4_way, get_average_distance
from dla_rand import DLA_init, plot_grid
import networkx as nx
import numpy as np
import pandas as pd
import os
import seaborn as sns

directory = "results/changing_stickness_400_walkers/"
#directory = "results/changing_walkers/"

# read the files, convert it to networks and get statistics

files = os.listdir(directory)
df = pd.DataFrame()

for file in files:
    print(file)
    try:
        grid = np.load(directory + file, allow_pickle = True)
    except OSError:
        continue
    # get simulation number and stickness
    stickiness = float(file[0:file.find(",")])
    simulation = (file[file.find(",")+1:file.find(".npy")])
    # convert grid to a network
    G = get_network(grid, simplify = True)
    average_degree = get_average_degree(G)
    average_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    diameter = nx.diameter(G)
    #radius = nx.radius(G)
    entropy = get_entropy(G, osmnx = False)
    dead_ends = get_dead_ends(G)
    ways_4 = get_4_way(G)
    # create a df dictionary
    df_new = pd.DataFrame({
        "stickiness" : [stickiness],
        "simulation" : [simulation],
        "average_degree" : [average_degree],
        "average_clustering" : [average_clustering],
        "transitivity" : [transitivity],
        "diameter": [diameter],
        #"radius" : [radius],
        "entropy" : [entropy],
        "dead_ends": [dead_ends],
        "ways_4" : [ways_4],
        "nodes": [len(G)],
        "nodes_diameter_ratio": [len(G)/diameter],
        "average_dsitance": get_average_distance(G)
        })
    df = df.append(df_new)
    
sns.lineplot(data = df, x = "stickiness", y = "average_dsitance")
    
#df.to_csv("results.csv")  