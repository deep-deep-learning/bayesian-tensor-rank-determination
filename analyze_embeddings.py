#%%
import torch

i = 2

fp = 'weights/emb%s.pt'%i

emb = torch.load(fp).T
print(emb.shape)

#%%
from sklearn.cluster import KMeans
import time
t = time.time()
predicted = KMeans(n_clusters = 200).fit_predict(emb.cpu().detach().numpy())
print("Total time ",time.time()-t)
#%%
200*250*250
