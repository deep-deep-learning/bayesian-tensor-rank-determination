#%%
import torch_bayesian_tensor_layers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch_bayesian_tensor_layers.low_rank_tensors
import torch
import torch_bayesian_tensor_layers.layers

#%%
from torch_bayesian_tensor_layers.layers import TensorizedEmbedding

#%%

shape = [[200, 200, 200], [4, 4, 2]]

prior_type = 'log_uniform'

max_rank = 10

# %%

tensor_type = 'TensorTrain'

emb = TensorizedEmbedding(shape=shape,
                          tensor_type=tensor_type,
                          max_rank=max_rank)

#print(torch.std(emb.tensor.sample_full()))

print(emb.tensor.target_stddev)

emb.to('cpu')

emb.tensor.get_kl_divergence_to_prior()

#%%

import numpy as np
import random
n = np.prod(shape[0])
batch_size = 1024
x_list = [random.randint(0, n - 1) for _ in range(batch_size)]
input_values = torch.tensor(x_list)

input_values = input_values.to('cuda')
from torch_bayesian_tensor_layers.emb_utils import get_tensorized_index
tensorized_indices = get_tensorized_index(input_values, emb.cum_prod)
rows = emb.forward(input_values.view(-1, 1))

#%%

i = 3
torch.std(emb.tensor.factors[i])

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
    cum_prod.append(cum_prod[-1] // x)

print(cum_prod)

indices = []

rem = input_values

for x in cum_prod:
    indices.append(rem // x)
    rem = torch.fmod(rem, x)

print(indices)

#%%
#%%

tmp = [mat.index_select(0, rows) for mat, rows in zip(factor_mats, indices)]

out = tl.kruskal_to_tensor((None, tmp + factor_mats[len(shape[0]):]))

out.shape

out = out.view([-1, np.prod(shape[1])])

#%%

for i in range(len(shape[0])):
    factor_mats[i] = torch.gather(factor_mats[i], )

#%%

out = tl.kruskal_to_tensor((None, factor_mats))
