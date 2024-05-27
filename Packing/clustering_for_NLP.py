import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
from plate_classifier import Plate_Classifier
import numpy as np
from datetime import datetime

def quater(month):
    if 1<=month<=3:
        return 0
    elif 4<=month<=6:
        return 1
    elif 7<=month<=9:
        return 2
    else:
        return 3

# Step 1: Read the CSV file into a pandas DataFrame
magnitudo = 6
df_org = pd.read_csv("Earthquakes.csv")
df = df_org[df_org['magnitudo']>=magnitudo]

f = open("event.csv",'w')
wr = csv.writer(f)


# Step 2: Extract latitude and longitude columns
coordinates = df[['longitude', 'latitude']]

# Step 3: Perform k-means clustering
k = 50  # Number of clusters

print(f"the operation on {k} clusters and {df.shape[0]} events with magnitudo {magnitudo} ")
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(coordinates)
cluster_centers=kmeans.cluster_centers_
pc = Plate_Classifier()

timestamps = df.time.values
timestamps_date = np.array(list(map(lambda x: datetime.fromtimestamp(int(x / 1000), tz=None), timestamps)))
first_year = timestamps_date[0].year
last_year = timestamps_date[-1].year #One month...

one_hot = [[False for i in range(k)] for j in range((last_year-first_year+1)*12)]
start_month = timestamps_date[0].month

for i in range(len(timestamps_date)):
    num_clst =  df.cluster.values[i]
    year = timestamps_date[i].year
    month = timestamps_date[i].month

    one_hot[(year-first_year)*12+month-1][num_clst] = True

print(one_hot[start_month-1])
for i in range(start_month-1,len(one_hot)):
    wr.writerow(one_hot[i])
