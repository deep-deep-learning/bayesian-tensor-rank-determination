#%%
import torch

i = 0

fp = 'weights/emb%s.pt'%i

emb = torch.load(fp)
#%%