import torch

def get_direction_mean_diff(pos_acts, neg_acts):
    """
    Basic steering: Mean difference between positive and negative activations.
    """
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)

def get_direction_pca(activations):
    """
    Advanced steering: Principal Component Analysis (Placeholder for future work).
    Returns the first principal component of the activation difference.
    """
    # Placeholder for future PCA implementation
    raise NotImplementedError("PCA steering not yet implemented.")
