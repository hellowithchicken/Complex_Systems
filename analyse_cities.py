from networks import get_city_network, get_network_stats
import pandas as pd
import networkx as nx


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
    df.to_csv(city + ".csv")
    
df["city"] = city_list

df.to_csv("city_results.csv")


