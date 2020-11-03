import torch
import numpy as np
import torch.nn as nn
from . import low_rank_tensors
from .low_rank_tensors import CP

class CPEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(CPEmbedding,self).__init__()

        self.shape = shape

        target_stddev = np.sqrt(2/np.prod(self.shape[0]))

        self.tensor = CP(self.shape[0]+self.shape[1],prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev)

        self.parameters = self.tensor.parameters

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx
        self.naive = naive
        self.cum_prod = torch.tensor(list(reversed([1]+list(np.cumprod(self.voc_quant[::-1])[:-1]))))

    
    def ind2sub(self,idx):

        out = []
        rem = idx

        for y in self.cum_prod.to(idx.device):
            val = torch.floor(rem.float() / y).long()
            rem = torch.fmod(rem, y)
    #        print(idx)

    #        val,rem = divmod(rem,y)
            out.append(val)

        out = torch.stack(out,dim=1).view(idx.shape[0],-1)
        return out

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        #x_ind = self.ind2sub(x)

        full = self.tensor.get_full()
        full = full.view([np.prod(self.shape[0]),np.prod(self.shape[1])])
        rows = full[x]

#        rows = gather_rows(self.tensor, x_ind)
        
        rows = rows.view(x.shape[0], -1)
        """         
        if self.naive:
            full = t3.naive_full(self.tt_matrix)
        else:
            full = self.tt_matrix.full()
        rows = full[x]
        """
        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training:
            self.tensor.update_rank_parameters()
        

        return rows.to(x.device)