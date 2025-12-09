import torch
import numpy as np


def dbscan(
    matrix: torch.Tensor,
    eps: float,
    min_samples: int,
) -> torch.Tensor:
    """
    DBSCAN density-based clustering on distance matrix.
    """
    if eps <= 0:
        raise ValueError("eps must be a positive value for DBSCAN.")

    device = matrix.device
    N = matrix.shape[0]

    # Convert to CPU Numpy for logic
    dist_mat = matrix.detach().cpu().numpy()

    # 1. Adjacency: dist <= eps
    adj = dist_mat <= eps

    # 2. Identify Core Points
    # count neighbors (includes self because dist(i,i)=0 <= eps)
    degrees = np.sum(adj, axis=1)
    core_mask = degrees >= min_samples

    labels = -1 * np.ones(N, dtype=np.int64)
    cluster_id = 0

    # 3. Cluster Expansion
    # Iterate over all points, but we only start expansion from unvisited core points
    for i in range(N):
        if labels[i] != -1 or not core_mask[i]:
            continue

        # Start new cluster
        labels[i] = cluster_id
        stack = [i]

        while stack:
            curr = stack.pop()
            neighbors = np.where(adj[curr])[0]

            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    # Was Noise or Unvisited -> assign to cluster
                    labels[neighbor] = cluster_id
                    if core_mask[neighbor]:
                        stack.append(neighbor)

        cluster_id += 1

    return torch.from_numpy(labels).to(device)


def dbscan2(
    matrix: torch.Tensor,
    eps: float,
    min_samples: int,
) -> torch.Tensor:
    """
    Performs a two-stage DBSCAN.
    1. Runs DBSCAN once.
    2. Runs DBSCAN again on the noise points from the first run.
    """
    # Run DBSCAN #1
    labels1 = dbscan(matrix, eps, min_samples)

    noise_indices_tensor = torch.where(labels1 == -1)[0]

    if len(noise_indices_tensor) < min_samples:
        # Not enough noise points to form a new cluster, return original labels
        return labels1

    # Create a sub-matrix for the noise points
    noise_matrix = matrix[noise_indices_tensor][:, noise_indices_tensor]

    # Find new eps for the noise points
    new_eps = find_dbscan_eps(noise_matrix, min_samples)

    # If no good eps is found, or too few points, we stop.
    if new_eps <= 0.0:
        return labels1

    # Run DBSCAN #2 on noise points
    labels2 = dbscan(noise_matrix, new_eps, min_samples)

    # Combine the results
    final_labels = labels1.clone()
    max_cluster_id = torch.max(labels1)

    # Re-label the clusters from the second run to be unique
    new_cluster_mask = labels2 != -1
    if torch.any(new_cluster_mask):
        # Offset new cluster IDs
        offset_labels2 = labels2[new_cluster_mask] + max_cluster_id + 1
        original_indices_of_new_clusters = noise_indices_tensor[new_cluster_mask]
        final_labels[original_indices_of_new_clusters] = offset_labels2
    return final_labels


def get_k_distances(matrix: torch.Tensor, k: int) -> np.ndarray:
    """
    Calculates the distance from each point to its k-th nearest neighbor.
    k: The rank of the nearest neighbor (e.g., k=1 means 2nd neighbor incl. self).
    Returns: Sorted k-distances (N,) numpy array.
    """
    if k < 1:
        return np.zeros(matrix.shape[0])

    dist_mat = matrix.detach().cpu().numpy()

    # Sort each row (distances from point i to all other points)
    # The k-th neighbor (including self at index 0) is at index k.
    sorted_dists = np.sort(dist_mat, axis=1)

    # k_distances is the distance to the (k+1)-th nearest neighbor (0-indexed k).
    k_distances = sorted_dists[:, k]

    # Sort the k-distances for the elbow plot
    k_distances.sort()

    return k_distances


def find_dbscan_eps(matrix: torch.Tensor, min_samples: int) -> float:
    """
    Finds the optimal epsilon for DBSCAN using the Kneedle algorithm heuristic
    on the k-distance graph (k = min_samples - 1).
    Heuristic: Maximum distance from the line connecting the first and last point.
    """
    N = matrix.shape[0]
    if N < 2:
        return 0.0

    k = max(1, min_samples - 1)
    k_distances = get_k_distances(matrix, k)

    x_coords = np.arange(N)
    y_coords = k_distances

    # Line equation: Ax + By + C = 0. A = y2 - y1, B = x1 - x2, C = -A*x1 - B*y1
    A = y_coords[-1] - y_coords[0]
    B = x_coords[0] - x_coords[-1]
    C = -A * x_coords[0] - B * y_coords[0]

    denominator = np.sqrt(A**2 + B**2)
    if np.isclose(denominator, 0.0):
        return np.median(y_coords)  # Flat line, use median

    numerator = np.abs(A * x_coords + B * y_coords + C)
    distances = numerator / denominator

    # Find the point with the maximum distance (the knee)
    knee_index = np.argmax(distances)

    return float(y_coords[knee_index])
