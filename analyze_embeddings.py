#%%
import numpy as np
import torch
import fastcluster
i = 3

fp = 'weights/emb%s.pt'%i

emb = torch.load(fp).T

emb = np.float16(emb.cpu().detach().numpy())
print(emb.shape)

#%%


#%%
from sklearn.cluster import KMeans
import time
t = time.time()
predicted = KMeans(n_clusters = 200).fit_predict(emb.cpu().detach().numpy())


print("Total time ",time.time()-t)

[200*250*250]

fp = 'weights/clusterings/emb%s'%i
import numpy as np
np.save(fp,predicted)
#%%

predicted = np.load('weights/clusterings/emb2.npy')


#%% Count number of entries by class
from collections import Counter
d = Counter(predicted) 

#%%
for x in range(100):
    print('{} has occurred {} times'.format(x, d[x])) 

# %%
