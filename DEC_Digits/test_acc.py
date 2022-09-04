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

X = torch.from_numpy(load_digits().data).float()
y = load_digits().target
n_clusters = len(np.unique(y))
print(X.shape, X.dtype, y.shape, n_clusters)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X = X.to(device)

# Initializing parameters

n_clusters          = 10
#batch_size          = 2000
#n_batches           = len(train_data)//batch_size
n_samples           = 1797
#train_loader        = DataLoader(train_data, batch_size = batch_size, shuffle = True)
#test_loader         = DataLoader(test_data, batch_size = batch_size, shuffle = True)
latent_size         = 10
dec                 = DEC_q2(latent_size).to(device)
mse                 = nn.MSELoss()
kl                  = nn.KLDivLoss(reduction="batchmean")
lr                  = 0.0001
noise_factor        = 0.1
n_pre_train         = 1000
n_train             = 10000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dec = DEC_q2(latent_size).to(device)

PATH = './saved_models/trained_exp_digits.pth'
dec.load_state_dict(torch.load(PATH))
dec.to(device)

centers     = dec.centers.clone().cpu().detach()
#print(torch.eq(b,centers))

# Extracting labels and latent from trained model

with torch.no_grad():
	z = dec.encoder(X).cpu()

# Calculating membership

dist = torch.cdist(z, centers)
pred = dist.argmin(axis=1).cpu()

for i in range(0,n_clusters,1):
  print((pred==i).sum())

# Testinng Accuracy, ARI and NMI values

y_test = y
k=n_clusters

# computation
true_classes=np.asarray(y_test)
pred_classes= pred.numpy()
no_correct=0
di={}

for i in range(k):
    di[i]={}
    for j in range(k):
        di[i][j]=[]
for i in range(true_classes.shape[0]):
    di[true_classes[i]][pred_classes[i]].append(1)
   
for i in range(len(di)):
    temp=-1
    for j in range(len(di[i])):
        temp=max(temp,len(di[i][j]))
        if temp==len(di[i][j]):
            cluser_class=j
    print("class {} named as class {} in clustering algo".format(list(di.keys())[i],cluser_class))
    no_correct=no_correct+temp
print(no_correct/true_classes.shape[0])

acc = no_correct/true_classes.shape[0]
ari = metrics.adjusted_rand_score(y, pred.numpy())
nmi = metrics.normalized_mutual_info_score(y, pred.numpy())
#t = end_time - start_time

print('Accuracy = {}'.format(acc))
print('ARI = {}'.format(ari))
print('NMI = {}'.format(nmi))

y = pred.cpu()
d = dec.cpu()
with torch.no_grad():
  rec_X = dec.encoder(X.cpu())
data = torch.vstack((rec_X, dec.centers))
print(data.shape)

from sklearn.manifold import TSNE
proj_X = TSNE(n_components=2).fit_transform(data.detach().cpu().numpy())

plt.figure(dpi=120)
plt.scatter(proj_X[0:-10,0], proj_X[0:-10,1], c=y, marker='x')
plt.scatter(proj_X[-10:,0], proj_X[-10:,1], c='r', marker='x', s=120)
plt.savefig('./results/TSNE_exp_1000_epochs_with_centers_digits.png', format = 'png')
