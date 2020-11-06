#%%
import os
import torch 
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
import torch.distributions as td
Parameter = torch.nn.Parameter
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch_bayesian_tensor_layers
#%%

from torch_bayesian_tensor_layers.low_rank_tensors import TensorTrainMatrix,Tucker,TensorTrain,CP


max_rank = 5
true_rank = 2
EM_STEPSIZE = 1.0


dims = [[10,10],[10,10]]

tensor = CP(dims=dims,max_rank=max_rank,prior_type='log_uniform',em_stepsize=EM_STEPSIZE)

true_tensor = TensorTrainMatrix(dims=dims,max_rank=true_rank,prior_type='log_uniform',em_stepsize=EM_STEPSIZE)


#%%
full = true_tensor.get_full().clone().detach()
#full = tl.tucker_to_tensor(tl.random.random_tucker(shape=dims,rank=true_rank))


log_likelihood_dist = td.Normal(0.0,0.001)


tensor.sample_full = tensor.get_full

def log_likelihood():
    return torch.mean(torch.stack([-torch.mean(log_likelihood_dist.log_prob(full-tensor.sample_full())) for _ in range(5)]))


def mse():
    return torch.norm(full-tensor.get_full())/torch.norm(full)

def kl_loss():
    return log_likelihood()+tensor.get_kl_divergence_to_prior()


loss = kl_loss

#loss = log_likelihood
#%%
optimizer = torch.optim.Adam(tensor.trainable_variables,lr=1e-2)

for i in range(10000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

    tensor.update_rank_parameters()

    if i%1000==0:
        print('Loss ',loss())
        print('RMSE ',mse())
        print('Rank ',tensor.estimate_rank())
        print(tensor.rank_parameters)

optimizer = torch.optim.Adam(tensor.trainable_variables,lr=1e-4)

for i in range(10000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

    tensor.update_rank_parameters()

    if i%1000==0:
        print('Loss ',loss())
        print('RMSE ',mse())
        print('Rank ',tensor.estimate_rank())
        print(tensor.rank_parameters)

#%%

i = 0
print(tensor.rank_parameters[i])
print(tensor.factor_distributions[1].stddev)
print(tensor)
#%%

print(tensor.factor_prior_distributions[-1].stddev[:,0,0])
print(tensor.rank_parameters[1])


#%%
tensor.update_rank_parameters()


#%%
import torch
import torch.distributions as td

mean = 0.0
std = torch.nn.Parameter(torch.tensor(1.0))

dist = td.Normal(mean,1.0)

dist.rsample()

def loss():
    return std*torch.norm(torch.relu(std)*dist.sample([100]))

# %%
optimizer = torch.optim.Adam([std],1e-2)


#%%

for _ in range(100):

    optimizer.zero_grad()
    loss_value = loss()
    loss_value.backward()
    optimizer.step()

    print('std',std.data.numpy())
    print(loss())

#%%
loss()

# %%

i = 0
j = 2
print(tensor.rank_parameters[i])
print(tensor.factor_prior_distributions[j].stddev[:,8,0])

#%%

print(tensor.factor_distributions[j].stddev[:,0,3,0])