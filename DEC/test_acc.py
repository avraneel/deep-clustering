import matplotlib.pyplot as plt
import numpy as np
import random 
import torch
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from k_means import kmeans
from dec_q2 import DEC_q2
from add_noise import addnoise
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

train_data = datasets.FashionMNIST(root='./data/FashonMNIST_data/', train=True, transform=transforms.ToTensor(), download=True)

test_data = datasets.FashionMNIST(root='./data/FashionMNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Initializing parameters

n_clusters          = 10
batch_size          = 2000
n_batches           = len(train_data)//batch_size
n_samples           = n_batches*batch_size
train_loader        = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader         = DataLoader(test_data, batch_size = batch_size, shuffle = True)
latent_size         = 10
dec                 = DEC_q2(latent_size).to(device)
mse                 = nn.MSELoss()
kl                  = nn.KLDivLoss(reduction="batchmean")
lr                  = 0.0001
noise_factor        = 0.1
n_pre_train         = 500
n_train             = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dec = DEC_q2(latent_size).to(device)

PATH = './saved_models/trained_q2.pth'
dec.load_state_dict(torch.load(PATH))
dec.to(device)

centers     = dec.centers.clone().cpu().detach()
#print(torch.eq(b,centers))

# Extracting labels and latent from trained model

latent_arr  = torch.zeros((n_batches, batch_size, latent_size))
labels_arr  = torch.zeros((n_batches, batch_size))
images_arr  = torch.zeros((n_batches, batch_size, 784))

for i,(images,labels) in enumerate(train_loader):
  with torch.no_grad():
    images, labels = images.to(device), labels.to(device)
    latent_arr[i] = dec.encoder(images).detach()
    labels_arr[i] = labels
    images_arr[i] = images.reshape((batch_size, 784))

arr_latent = latent_arr.reshape((n_samples,latent_size)).detach().cpu()
arr_labels = labels_arr.reshape(n_samples).detach().cpu().numpy()
arr_images = images_arr.reshape((n_samples, 784)).detach().cpu()

# Calculating membership


 
X = torch.from_numpy(load_digits().data).float()
y = load_digits().target
n_clusters = len(np.unique(y))

for i in range(0,n_clusters,1):
  print((y==i).sum())

print(y)


