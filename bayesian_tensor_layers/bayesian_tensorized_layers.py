import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3

from .low_rank_tensors import CP

class CPEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 prior_type='log_uniform',
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        self.shape = shape

        init = t3.glorot_initializer(self.shape, tt_rank=tt_rank)

        self.tensor = CP(self.shape[0]+self.shape[1],prior_type=prior_type)

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

        x_ind = self.ind2sub(x)

        full = self.tensor.get_full()
        full = full.view([np.prod(self.shape[0],np.prod(self.shape[1]))])
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

        return rows.to(x.device)