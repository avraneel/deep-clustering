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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Downloading data

#data

X = torch.from_numpy(load_digits().data).float()
y = load_digits().target
n_clusters = len(np.unique(y))
print(X.shape, X.dtype, y.shape, n_clusters)

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
n_train             = 1000

# Pre-training

optim       = torch.optim.Adam(dec.parameters(), lr)

start_time = time.time()

def pre_train(dec):

      pre_train_loss = []
      dec.train()

      for epoch in range(n_pre_train): 

        loss_epoch = 0

        image_noisy = addnoise(X,noise_factor)
        image_noisy.to(device)
        outputs = dec.decoder(dec.encoder(image_noisy))

        loss = mse(outputs, X) 

        optim.zero_grad()           
        loss.backward()                          
        optim.step()                

        loss_epoch += loss.item()

        mean_loss = loss_epoch/n_samples
        pre_train_loss.append(mean_loss)

        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, n_pre_train, mean_loss))
      
      print('Model pre-trained.\n')
      plt.xlabel('Total number of iterations')
      plt.ylabel('Loss')
      plt.plot(pre_train_loss)
      plt.savefig(fname='./results/pre_train_q2_digits.png', format='png')

pre_train(dec)

# Saving and Loading

PATH = './saved_models/pre_trained_q2_digits.pth'
torch.save(dec.state_dict(), PATH)

PATH = './saved_models/pre_trained_q2_digits.pth'
dec.load_state_dict(torch.load(PATH))
dec.to(device)

# Extracting all latent vectors from pre-trained model

with torch.no_grad():
  z = dec.encoder(X)

# Initializing centers by calling kmeans

dec.init_centers(z, n_clusters, device)
b = dec.centers.clone().cpu().detach()

# Training

optim       = torch.optim.Adam(dec.parameters(), lr)

def train(dec):

      train_loss = []
      dec.train()

      for epoch in range(n_train): 

        loss_epoch = 0

        image_noisy = addnoise(X,noise_factor)
        image_noisy.to(device)
        outputs = dec.decoder(dec.encoder(image_noisy))

        loss = mse(outputs, X) 

        optim.zero_grad()           
        loss.backward()                          
        optim.step()                

        loss_epoch += loss.item()

        mean_loss = loss_epoch/n_samples
        train_loss.append(mean_loss)

        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, n_train, mean_loss))
      
      print('Model pre-trained.\n')
      plt.xlabel('Total number of iterations')
      plt.ylabel('Loss')
      plt.plot(train_loss)
      plt.savefig(fname='./results/train_q2_digits.png', format='png')

train(dec)

end_time = time.time()

# Saving and Loading

PATH = './saved_models/trained_q2_digits.pth'
torch.save(dec.state_dict(), PATH)

PATH = './saved_models/trained_q2_digits.pth'
dec.load_state_dict(torch.load(PATH))
dec.to(device)

# Checking whether centers have updated

centers     = dec.centers.clone().cpu().detach()
print(torch.eq(b,centers))

# Extracting labels and latent from trained model

z_new = dec.encoder(X).cpu()

# Calculating membership

dist = torch.cdist(z_new, centers)
pred = dist.argmin(axis=1).cpu().detach()

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
print(no_correct/true_classes.shape[0]*100)

acc = no_correct/true_classes.shape[0]
ari = metrics.adjusted_rand_score(y, pred.numpy())
nmi = metrics.normalized_mutual_info_score(y, pred.numpy())
t = end_time - start_time

print('Accuracy = {}'.format(acc))
print('ARI = {}'.format(ari))
print('NMI = {}'.format(nmi))

with open("./results/q2_digits.txt", "a") as o:
    o.write('Number of epochs for pre-training = ' + str(n_pre_train))
    o.write("\n")
    o.write('Number of epochs for training = ' + str(n_train))
    o.write("\n")
    o.write('Accuracy = ' + str(acc))
    o.write("\n")
    o.write('ARI = ' + str(ari))
    o.write("\n")
    o.write('NMI = ' + str(nmi))
    o.write("\n")
    o.write('Time taken to execute = ' + str(t) + ' s')
    o.write("\n")
    o.write("\n")

# Plotting TSNE

proj_latent = TSNE(n_components=2, random_state=0).fit_transform(z_new.cpu().detach().numpy())

plt.figure(dpi=120)
plt.scatter(proj_latent[:,0], proj_latent[:,1], c = y)
plt.savefig(fname='./results/TSNE_q2_' + str(n_train) + '_epochs_digits.png', format='png')
plt.show()
