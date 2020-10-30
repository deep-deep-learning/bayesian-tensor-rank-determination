#%%

from layers import CPEmbedding
from low_rank_tensors import CP


shape = [[20,20],[10,10]] 

prior_type = 'log_uniform'

max_rank = 10

test = CP([10,10,10],max_rank=max_rank,prior_type=prior_type)


# %%

emb = CPEmbedding(shape=shape)

# %%
import torch
rows = torch.tensor([1,10,20])

a = emb(rows)
# %%
b = torch.reshape(emb.tensor.get_full(),[400,100])[rows]

#%%

a-b