import numpy as np
from scipy.spatial import distance

dataset = np.array([
    [1, 2, 3, 0],
    [2, 3, 1, 1],
    [3, 1, 2, 0],
    [4, 5, 1, 1],
    [3, 3, 4, 0]
])

new_observation = np.array([3, 3, 2])

def knn_classification(K):
    distances = []
    for idx, obs in enumerate(dataset):
        euclidean_distance = distance.euclidean(new_observation, obs[:3])
        distances.append((euclidean_distance, obs[3], idx))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:K]

    classes = [neighbor[1] for neighbor in nearest_neighbors]
    majority_class = max(set(classes), key=classes.count)

    return majority_class, nearest_neighbors

def knn_classification_weight(K):
    distances = []
    for idx, obs in enumerate(dataset):
        euclidean_distance = distance.euclidean(new_observation, obs[:3])
        distances.append((euclidean_distance, obs[3], idx))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:K]

    classes = [neighbor[1] for neighbor in nearest_neighbors]
    majority_class = max(set(classes), key=classes.count)

    return majority_class, nearest_neighbors


def print_nearest_neighbors(nearest_neighbors, K):
    print(f"\nThe nearest {K} neighbors for K={K}:")
    for neighbor in nearest_neighbors:
        distance, label, index = neighbor
        print(f"Neighbor Index: {index}, Distance: {distance:.4f}, Class Label: {label}")

class_k1, nearest_neighbors_k1 = knn_classification(1)
class_k3, nearest_neighbors_k3 = knn_classification(3)
class_k5, nearest_neighbors_k5 = knn_classification(5)

print(f"Assigned class for K=1: {class_k1}")
print(f"Assigned class for K=3: {class_k3}")
print(f"Assigned class for K=5: {class_k5}")

print_nearest_neighbors(nearest_neighbors_k1, 1)
print_nearest_neighbors(nearest_neighbors_k3, 3)
print_nearest_neighbors(nearest_neighbors_k5, 5)