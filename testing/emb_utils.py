
from torch_bayesian_tensor_layers.layers import TensorizedEmbedding
import torch
import numpy as np
import tensorly as tl
from functools import reduce

def cp_gather_rows(factors,tensorized_indices,shape):

    full_factors = factors

    tmp_factors = []

    for i,col in enumerate(tensorized_indices.unbind(1)):
        tmp_factors.append(full_factors[i][col,:])

    reduced = reduce(lambda x,y:x*y,tmp_factors)

    tmp_factors = [reduced]

    for factor in full_factors[-len(shape[1]):]:
        tmp_factors.append(factor)

    gathered_rows = tl.kruskal_to_tensor((None,tmp_factors)).view(-1,np.prod(shape[1]))

    return gathered_rows

def tt_reduce_fun(x,y):
    return torch.bmm(x,y.permute([1,0,2]))
#elif tensor_type =='TensorTrain':

def tt_gather_rows(cores,tensorized_indices,shape):

    tmp_cores = []
    for i,col in enumerate(tensorized_indices.unbind(1)):
        tmp_cores.append(cores[i][:,col,:])

    tmp_cores[0] = tmp_cores[0].permute([1,0,2])

    reduced = reduce(tt_reduce_fun,tmp_cores)

    tmp_factors = [reduced.permute([1,0,2])]

    for core in cores[-len(shape[1]):]:
        tmp_factors.append(core)


    batch_tensor = tl.tt_to_tensor(tmp_factors).view(-1,np.prod(shape[1]))

    return batch_tensor


def get_ttm_cum_prod(dims_0):

    out = []
    rem = idx

    cum_prod = [1]
    for x in reversed(dims_0[1:]):
        cum_prod.append(x*cum_prod[-1])

    cum_prod.reverse()
    cum_prod.pop()

    return cum_prod

def ttm_gather_rows(cores, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """

    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(core.shape[0]) for core in cores] + [1, ]

    for k, core in enumerate(cores):
        i = inds[:, k]
        cur_slice = torch.index_select(core, 1, i)
        # r x B x M x r

        if k == 0:
            res = cur_slice.transpose(0, 1)
            # B x r x M x r

        else:
            res = res.contiguous().view(batch_size, -1, ranks[k])
            # B x rM x r
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            # r x B x Mr
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))
    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))

    return res
"""
def convert_to_tt(idx,dims):
    out = []
    rem = idx

    for x in dims:
        val,rem = divmod(rem,int(x)) 
        out.append(val)
    return out
"""

def get_tensorized_index(x,cum_prod):
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

