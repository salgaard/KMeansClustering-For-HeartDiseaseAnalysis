import numpy as np
import pandas as pd


def LabelEncode(data):
    data['sex'] = data['sex'].astype('category').cat.codes
    data['dataset'] = data['dataset'].astype('category').cat.codes
    data['cp'] = data['cp'].astype('category').cat.codes
    data['fbs'] = data['fbs'].astype(int)
    data['restecg'] = data['restecg'].astype('category').cat.codes
    data['exang'] = data['exang'].astype(int)
    data['slope'] = data['slope'].astype('category').cat.codes
    data['thal'] = data['thal'].astype('category').cat.codes
    return data

def distance_to_center(centroids, data, assignments, cdist):
    distance = 0
    for c, centroid in enumerate(centroids):
        assigned_points = data[assignments == c, :]
        distance += np.sum(cdist(assigned_points, centroid.reshape(-1, 2)))
    return distance

def optimize_centroids(data, assignments):
    data_combined = np.column_stack((assignments.reshape(-1, 1), data))
    centroids = pd.DataFrame(data=data_combined).groupby(0).mean()
    return centroids.values

def assign_points(centroids, data, cdist):
    dist = cdist(data, centroids)        # all pairwise distances
    assignments = np.argmin(dist, axis=1)   # centroid with min distance
    return assignments

