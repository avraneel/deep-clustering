import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dec import DEC
from k_means import kmeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from add_noise import addnoise

#data

X = torch.from_numpy(load_digits().data).float()
y = load_digits().target
n_clusters = len(np.unique(y))
print(X.shape, X.dtype, y.shape, n_clusters)

denom = X.max(axis=0)[0] - X.min(axis=0)[0]
denom[denom==0] = 1
X = (X - X.min(axis=0)[0]) / denom
print(X.min(), X.max(), X.mean(axis=0).mean(), X.var(axis=0).mean())

# model initialization

dec1 = DEC(64, 50, 20, n_clusters)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X = X.to(device)
dec1.to(device)

print('--------------------------------------pre-traning----------------------------------')

max_epochs = 1000

loss = nn.MSELoss()
adam = optim.Adam(dec1.parameters())

for n_epoch in range(max_epochs):
    cost = torch.zeros(1).to(device)
    
    adam.zero_grad()
    z,_,_ = dec1(X + torch.normal(mean=torch.zeros(X.shape),std=torch.zeros(X.shape)+0.01).to(device))
    z = loss(z, X)
    with torch.no_grad():
        cost = z.clone()
    
    z.backward()
    
    adam.step()
    
    if n_epoch % 100 == 0 or (n_epoch+1)==max_epochs:
      print('epoch #'+str(n_epoch)+':', cost.item())

PATH = './pre_trained.pth'
torch.save(dec1.state_dict(), PATH)

print('-------------------------------pretraining done-------------------------')


PATH = './pre_trained.pth'
#model = model()
dec1.load_state_dict(torch.load(PATH))
dec1.to(device)

with torch.no_grad():
    latent_z = dec1.encoder(X)
dec1.init_centers(latent_z, n_clusters, device)
print(dec1.centers.shape)

print('-------------------------------training done-------------------------')
'''
max_epochs1 = 10000

mse_loss = nn.MSELoss()
kldiv_loss = nn.KLDivLoss(reduction='batchmean')
adam = optim.Adam(dec1.parameters())

for n_epoch in range(max_epochs1):
    cost = torch.zeros(1).to(device)
    
    adam.zero_grad()
    z, p, q = dec1(X + torch.normal(mean=torch.zeros(X.shape),std=torch.zeros(X.shape)+0.01).to(device))
    loss1 = mse_loss(z, X)
    loss2 = kldiv_loss(p, q)
    loss = loss1+loss2
    loss.backward()
    with torch.no_grad():
        cost = loss1.item()+loss2.item()
    
    adam.step()
    
    if n_epoch % 100 == 0 or (n_epoch+1)==max_epochs1:
      print('epoch #'+str(n_epoch)+':', cost)

PATH = './trained.pth'
torch.save(dec1.state_dict(), PATH)

print('-------------------------------training done-------------------------')

PATH = './trained.pth'
#model = model()
dec1.load_state_dict(torch.load(PATH))
dec1.to(device)

PATH = './trained.pth'
#model = model()
dec1.load_state_dict(torch.load(PATH))
dec1.to(device)

latent = dec1.encoder(X + torch.normal(mean=torch.zeros(X.shape),std=torch.zeros(X.shape)+0.01).to(device))

proj_X = TSNE(n_components=2).fit_transform(X.detach().cpu().numpy())
proj_latent = TSNE(n_components=2).fit_transform(latent.detach().cpu().numpy())

plt.figure(dpi=120)
plt.scatter(proj_X[:,0], proj_X[:,1], c=y)
plt.figure(dpi=120)
plt.scatter(proj_latent[:,0], proj_latent[:,1], c=y)
plt.show()
'''