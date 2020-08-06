#%%
import numpy as np

import matplotlib.pyplot as plt

n = 10000
r = 1
m = 1000

alpha = 0.1
beta = 0.1

X = 2*(-0.5+np.random.beta(alpha,beta,size=[n,r]))

for i in range(1,10):
    plt.figure()
    plt.hist(np.reshape(X,-1),bins=100)
    X*=i*2*(-0.5+np.random.beta(alpha,beta,size=[n,r]))
    

#%%

X = np.random.uniform(0,1,size=[n,r])
Y = np.random.uniform(0,1,size=[n,r])

alpha = 0.4
beta = alpha


X = np.random.beta(alpha,beta,size=[n,r])
Y = np.random.beta(alpha,beta,size=[n,r])

plt.hist(np.reshape(X+Y,-1),bins=100)
#%%
from scipy.stats import loguniform
X = loguniform.rvs(np.exp(-1/4),np.exp(1/4),size=[n,r])
Y = loguniform.rvs(np.exp(-1/4),np.exp(1/4),size=[r,m])
plt.hist(np.reshape(np.log(X)@np.log(Y),-1),bins=100)


# %%

import t3nsor as t3

shapes = [100,100,100]

n = 100*100*100
r = 2*4*4

layer = t3.TTEmbedding(voc_size=n,emb_size=r,auto_shapes=True)

for x in layer.tt_matrix.tt_cores:
    print(x.shape)

# %%
import torch
import random
import numpy as np
batch_size = 128

x_list = [random.randint(0,n) for _ in range(batch_size)]

x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [n, ]
x = x.view(-1)

tt_matrix = layer.tt_matrix

raw_shapes = tt_matrix.raw_shape


n_bases = [1]
m_bases = [1]

for k in range(1,len(raw_shapes[0])):
    n_bases.append(raw_shapes[0][-k]*n_bases[-1]) 
    m_bases.append(raw_shapes[1][-k]*m_bases[-1]) 

n_bases.reverse()
m_bases.reverse()
n_bases=torch.tensor(n_bases)
m_bases=torch.tensor(m_bases)


#%%


def convert_to_tt(idx,dims):

    dims = n_bases    
    out = []
    rem = idx

    for x in dims:
        val,rem = divmod(rem,int(x)) 
        out.append(val)


#%%


def a():
    x_list = [random.randint(0,n) for _ in range(batch_size)]
    full = t3.naive_full(tt_matrix)
    rows = full[x]

def b():
    x_list = [random.randint(0,n) for _ in range(batch_size)]
    full = tt_matrix.full()
    rows = full[x]

#%%
