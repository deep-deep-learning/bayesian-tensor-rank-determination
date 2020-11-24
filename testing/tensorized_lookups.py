#%%

import tensorly as tl
from torch_bayesian_tensor_layers.layers import TensorizedEmbedding
import t3nsor as t3
import torch
import random
import numpy as np


from emb_utils import ttm_gather_rows,get_tensorized_index,tt_gather_rows,cp_gather_rows


tensor_type = 'Tucker'

shape = [[10,50,20],[8,4]]

n = np.prod(shape[0])
r = np.prod(shape[1])

layer = TensorizedEmbedding(tensor_type=tensor_type,shape=shape)
factors = layer.tensor.factors
batch_size = 1024
x_list = [random.randint(0,n-1) for _ in range(batch_size)]
x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [n, ]
x = x.view(-1)
idx = x
full = layer.tensor.get_full() 
rows = layer.forward(idx)

print(idx)
dims = layer.shape[0]

out = []
rem = idx

cum_prod = [1]
for x in reversed(layer.shape[0][1:]):
    cum_prod.append(x*cum_prod[-1])

cum_prod.reverse()
cum_prod.pop()


tensorized_indices = get_tensorized_index(idx,cum_prod) 

if tensor_type == 'TensorTrainMatrix':
    gathered_rows = ttm_gather_rows(factors,tensorized_indices).view(batch_size,-1)
    print(torch.norm(rows-gathered_rows)/torch.norm(rows))
elif tensor_type == 'TensorTrain':
    gathered_rows = tt_gather_rows(factors,tensorized_indices,shape)
    print(torch.norm(rows-gathered_rows)/torch.norm(rows))
elif tensor_type =='CP':
    gathered_rows = cp_gather_rows(factors,tensorized_indices,shape)
    print(torch.norm(rows-gathered_rows)/torch.norm(rows))
elif tensor_type == 'Tucker':
    pass

#%%
print(idx)

print(tensorized_indices)

print(shape[0])


full_factors = layer.tensor.factors[1]
core = layer.tensor.factors[0]

tmp_factors = []
for i,col in enumerate(tensorized_indices.unbind(1)):
    print(col)
    tmp_factors.append(full_factors[i][col,:])

#%%

def reduce_fun(x,y):
    return torch.mult(x,y)

from functools import reduce

reduced = reduce(lambda x,y:x*y,tmp_factors)
print(reduced.shape)

tmp_factors = [reduced]

for factor in full_factors[-len(shape[1]):]:
    tmp_factors.append(factor)

gathered_rows = tl.kruskal_to_tensor((None,tmp_factors)).view(-1,np.prod(shape[1]))

"""

batch_tensor = tt_gather_rows(cores,tensorized_indices,layer.shape)

#batch_tensor = tl.tt_to_tensor(tmp_factors).view(-1,np.prod(shape[1]))

gathered_rows = batch_tensor

"""

torch.norm(rows-gathered_rows)/torch.norm(rows)

# %%
