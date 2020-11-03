#%%
import torch_bayesian_tensor_layers
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch_bayesian_tensor_layers.low_rank_tensors
from torch_bayesian_tensor_layers.low_rank_tensors import CP

import torch_bayesian_tensor_layers.layers
from torch_bayesian_tensor_layers.layers import CPEmbedding

#%%


shape = [[20,25,30],[10,10]] 

prior_type = 'log_uniform'

max_rank = 10

# %%

emb = CPEmbedding(shape=shape)

import torch
torch.std(emb.tensor.get_full())

#%%

kl_sum= 0.0

for p in emb.tensor.factor_distributions:
    var_ratio = (p.stddev / emb.tensor.rank_parameter).pow(2)
    t1 = ((p.mean ) / emb.tensor.rank_parameter).pow(2)
    kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))


#    term_1 =2*torch.log(a.stddev/emb.tensor.rank_parameter)

#    term_2 = (torch.square(a.stddev)+torch.square(a.mean))/(2*torch.square(emb.tensor.rank_parameter))

#    kl = torch.sum(term_1)+torch.sum(term_2)-a.stddev.numel()*0.5
    kl_sum+=kl
print(kl_sum)

print(emb.tensor.get_kl_divergence_to_prior())

#%%
emb.tensor.factor


# %%


input_values = torch.tensor([1,2,3])

rows = emb.forward(input_values)
# %%

tensor = emb.tensor

sample = True

if sample:
    factor_mats = [x.rsample() for x in tensor.factor_distributions]
else:
    factor_mats = tensor.factors

import tensorly as tl

#%%
import numpy as np
cum_prod = [np.prod(shape[0][1:])]

for x in shape[0][1:]:
    cum_prod.append(cum_prod[-1]//x)

print(cum_prod)

indices = []

rem = input_values

for x in cum_prod:
    indices.append(rem//x)
    rem = torch.fmod(rem,x)
    

print(indices)


#%%
#%%


tmp = [mat.index_select(0,rows) for mat,rows in zip(factor_mats,indices)]

out = tl.kruskal_to_tensor((None,tmp+factor_mats[len(shape[0]):]))


out.shape

out = out.view([-1,np.prod(shape[1])])

#%%


for i in range(len(shape[0])):
    factor_mats[i] = torch.gather(factor_mats[i],)



#%%

out = tl.kruskal_to_tensor((None,factor_mats))
