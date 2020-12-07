#%%
import torch_bayesian_tensor_layers
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch_bayesian_tensor_layers.low_rank_tensors
import torch
import torch_bayesian_tensor_layers.layers

from torch_bayesian_tensor_layers.layers import TensorizedEmbedding

shape = [[100, 100], [32]]

prior_type = 'log_uniform'

max_rank = 10

tensor_type = 'Tucker'

true_emb = TensorizedEmbedding(shape=shape,
                          tensor_type=tensor_type,
                          max_rank=max_rank//2)
trained_emb = TensorizedEmbedding(shape=shape,
                          tensor_type=tensor_type,
                          max_rank=max_rank)

#print(torch.std(emb.tensor.sample_full()))
lr = 1e-2

def get_batch_idx():
    n = np.prod(shape[0])
    x_list = [x for x in range(n)]
#    x_list = [random.randint(0, n - 1) for _ in range(batch_size)]
    input_values = torch.tensor(x_list)
    return input_values

true_emb.to('cuda')
trained_emb.to('cuda')

opt = torch.optim.Adam(lr=lr,params = trained_emb.parameters())


for ii in range(5000):

    opt.zero_grad()

    input_values = get_batch_idx().to(true_emb.tensor.factors[0].device)

    learned_rows = trained_emb.forward(input_values.view(-1, 1))

    true_rows = true_emb.forward(input_values.view(-1, 1))

    loss = torch.sum(torch.square(learned_rows-true_rows))

    kl_mult = 1e-5*torch.clamp(torch.tensor((ii-500)/500),torch.tensor(0.0),torch.tensor(1.0))

    loss+=kl_mult*trained_emb.tensor.get_kl_divergence_to_prior()

    loss.backward()

    opt.step()

    if ii%100==0:
        print(torch.sum(torch.square(learned_rows-true_rows)))
        print(trained_emb.tensor.estimate_rank())


#%%


full = trained_emb.tensor.get_full()

trained_emb.tensor.prune_ranks()

"""

factors = trained_emb.tensor.factors

rank_variables = trained_emb.tensor.rank_parameters

threshold = 1e-5

masks = [
            torch.tensor(torch.square(x)>threshold, dtype=torch.float32)
            for x in rank_variables
]
        

for mask in masks:
    print(mask.shape)
    print(mask)


factors = [x*y for x,y in zip(factors,masks)]+[masks[-1].view([-1,1,1,1])*factors[-1]]

"""

tmp = trained_emb.tensor.get_full()

print(torch.norm(tmp-full)/torch.norm(full))

tmp.shape
#%%

trained_emb.tensor.prune_ranks()


opt = torch.optim.Adam(lr=lr*1e-2,params = trained_emb.parameters())


for ii in range(1000):

    opt.zero_grad()

    input_values = get_batch_idx().to(true_emb.tensor.factors[0].device)

    learned_rows = trained_emb.forward(input_values.view(-1, 1))

    true_rows = true_emb.forward(input_values.view(-1, 1))

    loss = torch.sum(torch.square(learned_rows-true_rows))

    loss.backward()

    opt.step()

    if ii%100==0:
        print(torch.sum(torch.square(learned_rows-true_rows)))
        print(trained_emb.tensor.estimate_rank())


# %%
