#%%

from torch_bayesian_tensor_layers.layers import TensorizedEmbedding
import t3nsor as t3
import torch
import random
import numpy as np

def convert_to_tt(idx,dims):
    out = []
    rem = idx

    for x in dims:
        val,rem = divmod(rem,int(x)) 
        out.append(val)
    return out

def index_to_ttm_indices(x,cum_prod):
    rem = x
    out = []
    for x in cum_prod:
        val,rem = torch_divmod(rem,x) 
        out.append(val)

    out.append(rem)
    out = torch.stack(out).T
    return out


def torch_divmod(x,y):
    return torch.tensor(x)//torch.tensor(y),torch.fmod(torch.tensor(x),torch.tensor(y))


tensor_type = 'TensorTrainMatrix'

n = 2000
r = 4*4

layer = t3.TTEmbedding(voc_size=n,emb_size=r,shape=[[200,10],[4,4]])

batch_size = 20
x_list = [random.randint(0,n) for _ in range(batch_size)]
x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [n, ]
x = x.view(-1)
idx = x

tt_matrix = layer.tt_matrix

full = layer.tt_matrix.full() 

rows = full[idx]

print(idx)
dims = layer.shape[0]
out = []
rem = idx

cum_prod = [1]
for x in reversed(layer.shape[0][1:]):
    cum_prod.append(x*cum_prod[-1])

cum_prod.reverse()
cum_prod.pop()
print(cum_prod)

ttm_indices = index_to_ttm_indices(idx,cum_prod) 

print(layer.shape)
print(cum_prod)
x = 211
#idx = torch.tensor([x]).view(1,1)

#index_to_ttm_indices(idx,cum_prod)

"""
out = []
for x in cum_prod:
    val,rem = torch_divmod(rem,x) 
    out.append(val)

out.append(rem)
out = torch.stack(out).T
print(out)
tensorized_indices = torch.tensor(out).view(batch_size,-1)

"""


torch.norm(full[idx]-t3.gather_rows(layer.tt_matrix.cores,ttm_indices).view(batch_size,-1))/torch.norm(full[idx])

#%%











tensorized_indices = t3.ind2sub(layer.voc_quant,idx)

new_rows = t3.gather_rows(layer.tt_matrix,tensorized_indices)
new_rows = new_rows.view(idx.shape[0], -1)

torch.norm(rows-new_rows)/torch.norm(rows)
# %%
