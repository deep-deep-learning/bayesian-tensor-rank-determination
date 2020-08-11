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
import torch
import random
import numpy as np

batch_size = 20

x_list = [random.randint(0,n) for _ in range(batch_size)]
x = torch.tensor(x_list).view(batch_size,)

n = 100
r = 20

layer = t3.TTEmbedding(voc_size=n,emb_size=r,auto_shapes=True)

for core in layer.tt_matrix.tt_cores:
    print(core.shape)

#%%

std_out = layer(x)



def ind2sub(idx,dims):

    cum_prod =list(reversed([1]+list(np.cumprod(layer.voc_quant[1:]))))
    out = []
    rem = x.numpy()

    for i,y in enumerate(cum_prod):
        val,rem = divmod(rem,y)
        out.append(torch.tensor(val))

    out = torch.stack(out,dim=1).view(x.shape[0],-1)
    return out

x_ind = ind2sub(x,layer.voc_quant)

#x_ind = t3.ind2sub(layer.voc_quant, x)
#x_ind = out # torch.tensor([[0,0,1],[0,2,2]]).view([2,3])#t3.ind2sub(layer.voc_quant, x)
rows = t3.gather_rows(layer.tt_matrix, x_ind)
rows = rows.view(x.shape[0], -1)

torch.norm(rows-std_out)/torch.norm(std_out)

#%%
layer.voc_quant

cum_prod =list(reversed([1]+list(np.cumprod(layer.voc_quant[1:]))))

x

out = []
rem = x.numpy()

for i,y in enumerate(cum_prod):
    val,rem = divmod(rem,y)
    out.append(torch.tensor(val))

out = torch.stack(out,dim=1).view(x.shape[0],-1)


#%%
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
    out = []
    rem = idx

    for x in dims:
        val,rem = divmod(rem,int(x)) 
        out.append(val)
    return out

idx = 10101

tt_idx = convert_to_tt(idx,n_bases)

#%%

d = 3

full_out = tt_matrix.full()[idx]

my_out = t3.ops.gather_rows(tt_matrix,torch.reshape(torch.tensor(tt_idx),shape=(1,d)))


x_ind = t3.ind2sub(n_bases, idx)
rows = t3.gather_rows(self.tt_matrix, x_ind)
rows = rows.view(x.shape[0], -1)
#core_slices = 

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



import t3nsor as t3
import torch
import random
import numpy as np

shapes = [100,100]

n = 100
r = 2*4

layer = t3.TTEmbedding(voc_size=n,emb_size=r,auto_shapes=True)

for x in layer.tt_matrix.tt_cores:
    print(x.shape)

batch_size = 1

x_list = [random.randint(0,n) for _ in range(batch_size)]

x = torch.tensor(x_list)
xshape = list(x.shape)
xshape_new = xshape + [n, ]
x = x.view(-1)

tt_matrix = layer.tt_matrix

raw_shapes = tt_matrix.raw_shape

full = layer.tt_matrix.full()

rows = full[x]

tensorized_indices = t3.ind2sub(layer.voc_quant,x)

new_rows = t3.gather_rows(layer.tt_matrix,tensorized_indices)
new_rows = new_rows.view(x.shape[0], -1)

torch.norm(rows-new_rows)/torch.norm(rows)