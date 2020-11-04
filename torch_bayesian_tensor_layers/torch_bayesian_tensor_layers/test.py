
#%%
import torch 
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
from abc import abstractmethod, ABC
import torch.distributions as td
Parameter = torch.nn.Parameter
import numpy as np



#%%
from truncated_normal import TruncatedNormal

from low_rank_tensors import Tucker


max_rank = 5
true_rank = 2
EM_STEPSIZE = 1.0



dims = [50,50,50]
tensor = Tucker(dims=dims,max_rank=max_rank,prior_type='log_uniform',em_stepsize=EM_STEPSIZE)

full = tl.tt_to_tensor(tl.random.random_tt(shape=dims,rank=true_rank))


log_likelihood_dist = td.Normal(0.0,0.001)


def log_likelihood():
    return torch.mean(torch.stack([-torch.mean(log_likelihood_dist.log_prob(full-tensor.sample_full())) for _ in range(5)]))


def mse():
    return torch.norm(full-tensor.get_full())/torch.norm(full)

def kl_loss():
    return log_likelihood()+tensor.get_kl_divergence_to_prior()


loss = mse

#loss = log_likelihood

optimizer = torch.optim.Adam(tensor.trainable_variables,lr=1e-3)

#%%



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

print(tensor.factor_prior_distributions[-1].stddev[:,0,0])
print(tensor.rank_parameters[1])


#%%
tensor.update_rank_parameters()
