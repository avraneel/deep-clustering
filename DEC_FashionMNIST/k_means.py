import torch

def kmeans(X, k, max_iter=100, tol=1e-6):
    # Implement k-Means Clustering
    # 1. Initial step: Randomly initialize k cluster centers
    # 
    centers = X[torch.randperm(X.shape[0])[0:k]]

    for i2 in range(max_iter):
        # 2. (i) Calculate the distance of all data points to all cluster centers
        dist = torch.cdist(X, centers)

        # 2. (ii) For each data point, which cluster center lies closest to it
        mem = dist.argmin(axis=1)

        # 2. (iii) Recompute our cluster centers, as the mean of the data points that are closest to it
        prev_centers = centers.clone().detach()
        for j in range(k):
          centers[j] = X[mem==j].mean(axis=0)

        if torch.norm(centers - prev_centers) < tol:
          print('break at iter', i2)
          break

    return centers, mem