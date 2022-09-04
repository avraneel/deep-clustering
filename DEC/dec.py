from asyncio.unix_events import _UnixDefaultEventLoopPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from k_means import kmeans

class DEC(nn.Module):
    def __init__(self, input_dim, h1_dim_size, h2_dim_size, h3_dim_size, latent_size, n_clusters):
        super(DEC, self).__init__()
        self.flatten = nn.Flatten()
        self.enc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, h1_dim_size)           # 784->500
        )
        self.enc2 = nn.Linear(h1_dim_size, h2_dim_size)         # 500->500
        self.enc3 = nn.Linear(h2_dim_size, h3_dim_size)         # 500->2000
        self.enc4 = nn.Linear(h3_dim_size, latent_size)         # 2000->10
        self.dec1 = nn.Linear(latent_size, h3_dim_size)         # 10->2000
        self.dec2 = nn.Linear(h3_dim_size, h2_dim_size)         # 2000->500
        self.dec3 = nn.Linear(h2_dim_size, h1_dim_size)         # 500->500
        self.dec4 = self.enc1 = nn.Sequential(
            nn.Linear(input_dim, h1_dim_size),           # 500->784
            nn.Unflatten(dim=1, unflattened_size= (1,28,28))
        )           
        self.centers = nn.Parameter(torch.empty((n_clusters, latent_size)))

    def init_centers(self, z, n_clusters, device):
        i_centers, _ = kmeans(z,n_clusters)
        self.centers = nn.Parameter(i_centers)
        #km1 = KMeans(n_clusters=n_clusters).fit(z.detach().cpu().numpy())
        #self.centers = nn.Parameter(centers).float().to(device))

    def encoder(self, x):
        z = F.relu(self.enc1(x))
        z = F.relu(self.enc2(z))
        z = F.relu(self.enc3(z))
        z = F.relu(self.enc4(z))
        return z

    def clustering(self, z):
        q = 1 / (1 + (torch.cdist(self.centers, z, p=2) ** 2))
        q = (q / q.sum(axis=0)).T # nxk
        p = (q ** 2 / q.sum(axis=0)).T
        p = (p / p.sum(axis=0)).T # nxk
        return p, q

    def decoder(self, x):
        z = F.leaky_relu(self.dec1(x))
        z = F.leaky_relu(self.dec2(z))
        z = F.leaky_relu(self.dec3(z))
        z = F.leaky_relu(self.dec4(z))
        return z
    
    def forward(self, x):
        z = F.relu(self.enc1(x))
        z = F.relu(self.enc2(z))
        z = F.relu(self.enc3(z))
        z = F.relu(self.enc4(z))
        p, q = self.clustering(z)
        z = F.leaky_relu(self.dec1(z))
        z = F.leaky_relu(self.dec2(z))
        z = F.leaky_relu(self.dec3(z))
        z = F.leaky_relu(self.dec4(z))
        return z, p ,q
