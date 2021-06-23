from networks import get_network_stats, get_city_network
import pandas as pd

city_list = ["Bangkok, Thailand", "Manila, the Philippines", "Ulaanbaatar, Mongolia", 
             "Melbourne, Australia",
             "Oslo, Norway", "Amsterdam, the Netherlands", "Kaunas, Lithuania",
             "Las Vegas, NV, USA", "Phoenix, AZ, USA", "Monterrey, Mexico",
             "Manaus, Brazil", "Asunción, Paraguay", 
             "Lagos, Nigeria", "Kampala, Uganda", 
             "Riyadh, Saudi Arabia", "Sana'a, Yemen"]


df = pd.DataFrame()

for city in city_list:
    print(city)
    G = get_city_network(city)
    stats_df = get_network_stats(G, osmnx = True)
    df = df.append(stats_df)
    df.to_csv(city + ".csv")
    
df["city"] = city_list

df.to_csv("city_results.csv")


