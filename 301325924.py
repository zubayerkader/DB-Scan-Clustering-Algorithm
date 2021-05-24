import pandas as pd
import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def dist (row, k, data):
    dist_col = data.sub(row, axis=1).pow(2).sum(axis=1).pow(.5)
    knn_dist = dist_col.sort_values(ascending=True).head(k+1)
    furthest_dist = knn_dist.sort_values(ascending=False).reset_index()
    dist = furthest_dist.iloc[0][0]
    return dist

def k_distance(data, k=None):
    if k is None:
        k = 2*7-1       # from lecture slide 271
    df = data.copy()
    df["dist_to_furthest_kneighbor"] = df.apply(dist, axis=1, k=k, data=df)
    df = df.sort_values(by="dist_to_furthest_kneighbor", ascending=False)
    plt.plot(list(df["dist_to_furthest_kneighbor"]), '.')
    plt.savefig(str(k) + '-distance-diagram.jpg')

def core_object(row, data, E, min_pts):
    df = data
    if "cluster label" in df.columns:
        df = data.drop(columns=["cluster label"])        
    dist_col = df.sub(row, axis=1).pow(2).sum(axis=1).pow(.5)
    neighbors = dist_col[dist_col <= E]
    return neighbors.shape[0] >= min_pts

def assign_cluster_label(row, cluster_id):
    cluster_label = row["cluster label"]
    if cluster_label == str(-1):
        return str(cluster_id)
    else:
        label_list = str(cluster_label).split(",")
        # check for duplicate cluster_id before appending
        if str(cluster_id) in label_list:
            return cluster_label
        else:
            # append cluster_id to neighbors
            label_list.append(str(cluster_id))
            listToStr = ','.join([str(elem) for elem in label_list]) 
            return listToStr


def density_reachable(data, row, cluster_id, E, min_pts):
    frontier = pd.DataFrame([], columns=data.columns)
    explored = pd.DataFrame([], columns=data.columns)
    frontier = frontier.append(row)
    df = data.drop(columns=["cluster label"])
    while not frontier.empty:
        first_row = frontier.iloc[0]
        print ("first_row", first_row)

        print("frontier before popping core obj", frontier)
        frontier = frontier.iloc[1:]
        print ("frontier after popping core obj", frontier)

        explored = explored.append(first_row)
        print ("explored", explored)

        print("original data['cluster label'] before", data.loc[first_row.name]["cluster label"])
        data["cluster label"].loc[first_row.name] = assign_cluster_label(first_row, cluster_id)     ##### check if data has been assigned cluster labels after this function is over
        print("original data['cluster label'] after", data.loc[first_row.name]["cluster label"])

        # get neighbors of first row
        dist_col = df.sub(row, axis=1).pow(2).sum(axis=1).pow(.5)
        neighbors = dist_col[dist_col <= E]     # contains indices and distance of neighbors
        neighbors = neighbors.sort_values(ascending=True)
        neighbors = neighbors.iloc[1:]          # remove first_row from neighbors
        print("neighbors", neighbors)

        for n in neighbors.index:     
            if n not in frontier.index and n not in explored.index:
                neighbor = data.loc[n]
                print("neighbor", neighbor)
                if core_object(neighbor, data, E, min_pts):
                    print("frontier before appending neighbor", frontier)
                    frontier = frontier.append(data.loc[n])
                    print("frontier after appending neighbor", frontier)
                else:
                    print("original data['cluster label'] before", data["cluster label"].loc[n])
                    data["cluster label"].loc[n] = assign_cluster_label(neighbor, cluster_id)
                    print("original data['cluster label'] after", data["cluster label"].loc[n])

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! processed 1 core obj from frontier and its epsilon neighborhood")

# DBSCAN Algorithm
def fit(data, E=None, min_pts=None):
    if E is None:
        E = 0.5
    if min_pts is None:
        min_pts = 9
    data["cluster label"] = str(-1)
    cluster_list = []
    cluster_id = 0
    for index, row in data.iterrows():
        if row["cluster label"] == str(-1):
            if core_object(row, data, E, min_pts):
                # print ("core object")
                density_reachable(data, row, cluster_id, E, min_pts)
                cluster_list.append(cluster_id)
                cluster_id += 1

    data.to_csv("./cluster_labeled_data.csv")
    print (cluster_list)
    return cluster_list

def preprocess(data):
    # impute
    data.replace('?', value=None, inplace=True) # interpolates missing values using default method='pad'
    # remove date and time column
    datetime_col = data[["Date", "Time"]]
    data = data.drop(columns=["Date", "Time"])
    # normmalize
    std_scaler = StandardScaler()
    data = pd.DataFrame(std_scaler.fit_transform(data), columns=data.columns)
    # data = datetime_col.merge(data, how="inner", left_index=True, right_index=True)
    return data

def num_objects_in_cluster(cluster_list):
    clustered_data = pd.read_csv("./cluster_labeled_data.csv")
    objects_in_cluster = {}
    for c in cluster_list:
        object_count = 0
        for i, row in clustered_data.iterrows():
            cluster_label = row["cluster label"]
            label_list = str(cluster_label).split(",")
            if str(c) in label_list:
                object_count+=1
        objects_in_cluster[str(c)] = object_count
    return objects_in_cluster

def main(filename):
    print("working...")
    data = pd.read_csv(filename)
    data = data[data['Date'].apply(lambda x: x.endswith('/1/2007'))]
    processed_data = preprocess(data)

    # print("Generating k-distance-diagram...")
    # start = time.time()
    # k_distance(processed_data)
    # print(processed_data)
    # end = time.time()
    # print ("total time for k-distance-diagram", end-start)

    print("Clustering Data...")
    E = 0.6      # from k-distance-diagram
    min_pts = 2*7     # from lecture slide 271
    start = time.time()
    cluster_list = fit(processed_data, E, min_pts)
    end = time.time()
    print ("total time for clustering", end-start)

    print("{'cluster': number of objects in cluster}\n", num_objects_in_cluster(cluster_list))

    print("Done")

if __name__ == "__main__":
    main("./houshold2007.csv")