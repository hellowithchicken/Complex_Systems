from networks import get_city_network, get_average_degree, get_entropy, get_dead_ends, get_4_way  #, get_network_stats
import pandas as pd
import networkx as nx


def get_network_stats(G, osmnx = False):
  """
  Takes graph G and returns a dataframe with selected network statistics
  Set osmnx = True if the graph was obtained via osmnx as networks from osmnx package 
  have easy entropy calculation option
  """
  average_degree = get_average_degree(G)
  average_clustering = nx.average_clustering(G)
  transitivity = nx.transitivity(G)
  diameter = nx.diameter(G)
  entropy = get_entropy(G, osmnx)
  dead_ends = get_dead_ends(G)
  ways_4 = get_4_way(G)
  # create a df dictionary
  if osmnx == False:
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
        #"nodes_diameter_ratio": [len(G)/diameter],
        #"average_distance" : [average_distance]
        })
  else:
    df = pd.DataFrame({
        "average_degree" : [average_degree],
        "average_clustering" : [average_clustering],
        "transitivity" : [transitivity],
        "entropy" : [entropy],
        "dead_ends": [dead_ends],
        "ways_4" : [ways_4],
        "diameter" : [diameter],
        "nodes": [len(G)],
        "nodes/diameter": [len(G)/diameter]
        })
  return df

city_list = ["Kaunas, Lithuania", "Bangkok, Thailand", "Manila, the Philippines", "Ulaanbaatar, Mongolia", 
             "Melbourne, Australia",
             "Oslo, Norway", "Amsterdam, the Netherlands",
             "Las Vegas, NV, USA", "Phoenix, AZ, USA", "Monterrey, Mexico",
             "Manaus, Brazil", "Asunción, Paraguay", 
             "Lagos, Nigeria", "Kampala, Uganda", 
             "Riyadh, Saudi Arabia", "Sana'a, Yemen", "Berlin, Germany"]


df = pd.DataFrame()

for city in city_list:
    print(city)
    G = get_city_network(city)
    stats_df = get_network_stats(G, osmnx = True)
    df = df.append(stats_df)
    df.to_csv(city + "2.csv")
    
df["city"] = city_list

df.to_csv("city_results_2.csv")


