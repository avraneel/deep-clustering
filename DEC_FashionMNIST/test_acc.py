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

train_data = datasets.FashionMNIST(root='./data/FashionMNIST_data/', train=True, transform=transforms.ToTensor(), download=True)

test_data = datasets.FashionMNIST(root='./data/MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

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

PATH = './old/saved_models 1/trained_exp_fashion.pth'
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

dist = torch.cdist(arr_latent, centers)
pred = dist.argmin(axis=1).cpu()

for i in range(0,n_clusters,1):
  print((pred==i).sum())

# Testinng Accuracy, ARI and NMI values

y_test = arr_labels
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
ari = metrics.adjusted_rand_score(arr_labels, pred.numpy())
nmi = metrics.normalized_mutual_info_score(arr_labels, pred.numpy())
#t = end_time - start_time

print('Accuracy = {}'.format(acc))
print('ARI = {}'.format(ari))
print('NMI = {}'.format(nmi))

X = arr_images
y = pred.cpu()
d = dec.cpu()
with torch.no_grad():
  rec_X = d.encoder(X)
data = torch.vstack((rec_X, dec.centers))
print(data.shape)

from sklearn.manifold import TSNE
proj_X = TSNE(n_components=2).fit_transform(data.detach().cpu().numpy())

plt.figure(dpi=120)
plt.scatter(proj_X[0:-10,0], proj_X[0:-10,1], c=y, marker='x')
plt.scatter(proj_X[-10:,0], proj_X[-10:,1], c='r', marker='x', s=120)
plt.savefig('./results/TSNE_exp_500_epochs_with_centers_2_fashion.png', format = 'png')
