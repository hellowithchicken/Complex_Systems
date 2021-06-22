import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd

### CHECK betweenness_centrality LATER

def check_for_neighbours(row, col, grid, direction):
  """
  Looks in a specified direction for a neighbour
  """
  if direction == "up" and (row - 1 > -1):
    return grid[row - 1, col] > 0
  if direction == "down" and (row + 1 < len(grid)):
    return grid[row + 1, col] > 0
  if direction == "right" and (col + 1 < len(grid)):
    return grid[row, col + 1] > 0
  if direction == "left" and (col - 1 > -1):
    return grid[row, col - 1] > 0
  return False

def get_network(grid, simplify = True, plot = False):
  """
  Takes a grid of 1s and 0s and converts it into a network:
      - goes over each cell in the grid and connects links between occupied
      cells to the east, west, nort and south.
  Simplify - removed nodes of degree 2 but keeps the links.
  GRID HAS TO BE A SQUARE.
  """
  # initialise network
  G=nx.Graph()

  node_number = 0
  nodes = []
  links = []
  pos = {}
  for row in range(len(grid)):
    for col in range(len(grid)):
      # update node number
      node_number += 1
      # check if there is anything in cell - if there is not, skip it completely to save time
      if grid[row][col] == 0:
        continue
      # add node to the list
      nodes.append(node_number)
      # get coordinates of the node
      coordinates  = (col, -1 * row)
      pos[node_number] = coordinates
      # check for neighbors and if the neighbor exists add a link
      # check up
      if check_for_neighbours(row, col, grid, "up"):
        links.append((node_number, node_number - len(grid)))
      # check down
      if check_for_neighbours(row, col, grid, "down"):
        links.append((node_number, node_number + len(grid)))
      # check left
      if check_for_neighbours(row, col, grid, "left"):
        links.append((node_number, node_number - 1))
      # check down
      if check_for_neighbours(row, col, grid, "right"):
        links.append((node_number, node_number + 1))


  # add nodes to the graph
  G.add_nodes_from(nodes)
  # add links to the graph
  G.add_edges_from(links)


  # remove nodes of degree 2
  if simplify:
    for node in list(G.nodes()):
      if G.degree(node) == 2:
          edges = list(G.edges(node))
          G.add_edge(edges[0][1], edges[1][1])
          G.remove_node(node)

  # add position (coordinate) attribute to each node
  nx.set_node_attributes(G, pos, "pos")

  # produce a plot
  if plot:
      nx.draw(G, pos)
      
  return G

def get_city_network(city, simplify = True, get_multi = False):
  """
  Returns a city network from OSMnx package.
  The function downloads the city network and returns the giant component
  to make sure that the network is fully connected.
  simplify - returns a graph that only has junctions and dead-end.
  city - string name of the city e.g. "Amsterdam, the Netherlands",
  get_multi - set to True if you will be using the graph for plotting with plot_city_network
  , this way you get two graphs - G: suitable for statistics, G0 - suitable for plotting
  """
  G = ox.graph_from_place(city, network_type="drive", simplify = simplify)
  G = ox.bearing.add_edge_bearings(G) # add bearings - needed for entropy calculations
  # convert to undirected
  G = G.to_undirected()
  # get the giant component
  Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
  G0 = G.subgraph(Gcc[0])
  G = nx.Graph(G)
  if get_multi:
    return G, G0
  return G

def get_degree_list(G):
  """
  Takes graph G and returns list of node degress
  """
  degrees = G.degree()
  degree_list = []
  for node in degrees:
    degree_list.append(node[1])
  return degree_list

def get_average_degree(G):
  """
  Takes a graph G and returns average degree of the graph
  """
  degrees = G.degree()
  no_of_nodes = len(G)
  sum = 0
  for node in degrees:
    sum += node[1]
  return sum/no_of_nodes


def get_dead_ends(G):
  """
  Takes a graph G and returns the fraction of dead-end streets (degree = 1)
  """
  degrees = G.degree()
  no_of_nodes = len(G)
  sum = 0
  for node in degrees:
    if node[1] == 1:
      sum += 1
  return sum/no_of_nodes

def get_4_way(G):
  """
  Takes graph G and returns the fraction of 4way intersections (degree = 4)
  """
  degrees = G.degree()
  no_of_nodes = len(G)
  sum = 0
  for node in degrees:
    if node[1] == 4:
      sum += 1
  return sum/no_of_nodes

def get_entropy(G, osmnx = False):
  """
  Takes graph G and returns its' orientational entropy based on (Shannon 1948).
  Set osmnx = True if the graph was obtained via osmnx as networks from osmnx package 
  have easy entropy calculation option
  """
  if osmnx:
    return ox.bearing.orientation_entropy(G)
  else:
    # calculate bearings (the angle of the edge with the north)
    coordinates=nx.get_node_attributes(G,'pos')
    bearings = []
    for e in G.edges:
      node_1= e[0]
      node_2= e[1]
      x_1, y_1 = coordinates[node_1]
      x_2, y_2 = coordinates[node_2]
      bearing = ox.bearing.calculate_bearing(x_1, y_1, x_2, y_2)
      bearings.append(bearing)
  #bin the bearings into 36 bins, meaning 10 angles in each bin
  bins = np.linspace(0, 360, 37)
  bin_indices = np.digitize(bearings, bins)
  entropy = 0
  bin_sizes = []
  for n in np.linspace(1,36,36):
      bin_size = 0
      for b in bin_indices:
          if n == b:
              bin_size += 1
      fraction = bin_size/len(bin_indices)
      if fraction > 0:
          entropy += fraction * np.log(fraction)
  return -1 * entropy

def get_network_stats(G, osmnx = False):
  """
  Takes graph G and returns a dataframe with selected network statistics
  Set osmnx = True if the graph was obtained via osmnx as networks from osmnx package 
  have easy entropy calculation option
  """
  average_degree = get_average_degree(G)
  average_clustering = nx.average_clustering(G)
  transitivity = nx.transitivity(G)
  #diameter = nx.diameter(G)
  #radius = nx.radius(G)
  entropy = get_entropy(G, osmnx)
  dead_ends = get_dead_ends(G)
  ways_4 = get_4_way(G)
  # create a df dictionary
  df = pd.DataFrame({
      "average_degree" : [average_degree],
      "average_clustering" : [average_clustering],
      "transitivity" : [transitivity],
      #"diameter": [diameter],
      #"radius" : [radius],
      "entropy" : [entropy],
      "dead_ends": [dead_ends],
      "ways_4" : [ways_4],
      "nodes": [len(G)],
      #"nodes_diameter_ratio": [len(G)/diameter]
      })
  return df

def plot_city_network(G):
  """
  Takes a city graph G and plots it.
  G must be of networkx.MultiDiGraph - so you must set get_multi = True
  in get_city_network function to be able to plot.
  """
  ox.plot_graph(
    G,
    show=False,
    close=False,
    bgcolor="#333333",
    edge_color="b",
    edge_linewidth=0.3,
    node_size=1,
  )

def plot_grid_network(G, fig_size = 10, node_size = 25, node_color = "#008494"):
    """
    Takes network G (generated with DLA) and plots it on a square plot.
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.axis('off')

    nx.draw_networkx(G, 
                     pos = nx.get_node_attributes(G,'pos'),
                     with_labels=False,
                     node_color = node_color,
                     node_size = node_size,
                     ax = ax)


