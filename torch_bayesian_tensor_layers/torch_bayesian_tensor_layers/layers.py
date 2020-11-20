import torch
import numpy as np
import torch.nn as nn
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from . import low_rank_tensors


class TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(2/np.prod(self.shape[0]))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            tensor_shape = self.shape[0]+self.shape[1]

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False)

        self.parameters = self.tensor.parameters

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        #x_ind = self.ind2sub(x)

        full = self.tensor.sample_full()
        full = torch.reshape(full,[self.voc_quant,self.emb_quant])
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
