import numpy as np
# Given list of weight vectors
weight_vectors_new = np.array([
    [0.181818, 0.454545, 0.363636],
    [0.181818, 0.454545, 0.363636],
    [0.181818, 0.454545, 0.363636],
    [0.181818, 0.454545, 0.363636],
    [0.181818, 0.454545, 0.363636],
    [0.166667, 0.458333, 0.375],
    [0.153846, 0.461538, 0.384615]
])

# Compute the mean consensus vector
consensus_vector_new = np.mean(weight_vectors_new, axis=0)

# Compute the distances of each expert's weight vector from the consensus vector
distances_new = np.linalg.norm(weight_vectors_new - consensus_vector_new, axis=1)

# Compute the final aggregated distance as the mean of all distances
final_distance_new = np.mean(distances_new)

# Display the final distance
print(final_distance_new)
