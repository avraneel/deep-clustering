import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from k_means import kmeans

n_clusters = 10

class DEC_q2(nn.Module):

  def __init__(self, latent_size):
    super(DEC_q2, self).__init__()
    
    self.encoder = nn.Sequential(
        nn.Flatten(),
        
        nn.Linear(784, 500),
        nn.ReLU(),

        nn.Linear(500,500),
        nn.ReLU(),          

        nn.Linear(500,2000),
        nn.ReLU(),

        nn.Linear(2000,latent_size),
        nn.ReLU()
    )

    self.centers = nn.Parameter(torch.empty((n_clusters, latent_size)))

    self.decoder = nn.Sequential(
      nn.Linear(latent_size, 2000),
      nn.ReLU(),

      nn.Linear(2000, 500),
      nn.ReLU(),          

      nn.Linear(500, 500),
      nn.ReLU(),

      nn.Linear(500, 784),
      nn.ReLU(),

      nn.Unflatten(dim=1, unflattened_size= (1,28,28))
  )

  def clustering(self, z):
    q = 1 / (1 + (torch.cdist(self.centers, z, p=2) ** 2))
    q = (q / q.sum(axis=0)).T # nxk
    p = ((q**2) / q.sum(axis=0)).T
    p = (p / p.sum(axis=0)).T # nxk
    return p, q

  def init_centers(self, z, n_clusters, device):
      i_centers, _ = kmeans(z,n_clusters)
      self.centers = nn.Parameter(i_centers.float().to(device))
      #km1 = KMeans(n_clusters=num_clusters).fit(z.detach().cpu().numpy())
      #self.centers = nn.Parameter(torch.from_numpy(km1.cluster_centers_).float().to(device))

  def forward(self, x):
    x = self.encoder(x)
    latent_space = x
    p, q = self.clustering(x)
    x = self.decoder(x)

    return x, latent_space, p, q