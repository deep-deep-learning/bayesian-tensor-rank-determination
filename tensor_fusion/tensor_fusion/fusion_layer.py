import torch
import torch.nn as nn
import numpy as np
from .low_rank_tensor import CP_with_trainable_rank_parameter

class AdaptiveRankFusionLayer(nn.Module):

    def __init__(self, input_sizes, output_size, bias=False,
                 max_rank=10, prior_type='log_uniform', eta=None, 
                 device=None, dtype=None):
        '''
        args:
            input_sizes: a tuple of ints, (input_size_1, input_size_2, ..., input_size_M)
            output_sizes: an int, output size of the fusion layer
            max_rank: an int, maximum rank for the CP decomposition
            eta: a float, hyperparameter for rank parameter distribution
            device:
            dtype:
        '''
        super(AdaptiveRankFusionLayer, self).__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.max_rank = max_rank
        self.eta = eta
        self.device = device
        self.dtype = dtype

        tensorized_shape = input_sizes + (output_size,)
        #target_stddev = np.sqrt(2/np.prod(input_sizes))
        self.weight_tensor = CP_with_trainable_rank_parameter(tensorized_shape, 
                                                              prior_type=prior_type, 
                                                              max_rank=max_rank, 
                                                              initialization_method='nn',
                                                              #target_stddev=target_stddev,
                                                              learned_scale=False,
                                                              eta=eta,
                                                              device=device,
                                                              dtype=dtype)
        
        self.weight_tensor.to(dtype)
        self.weight_tensor.to(device)

        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, inputs):
        '''
        args:
            inputs: a list of vectors, (input_1, input_2, ..., input_M)
        return:
            y = [(input_1 @ factor_1) (input_2 @ factor_2) ... (input_M @ factor_M)] @ factor_{M+1}.T
        '''

        y = 1.0
        for i, x in enumerate(inputs):
            y = y * (x @ self.weight_tensor.factors[i])
        y = y @ self.weight_tensor.factors[-1].T

        if self.bias is not None:
            y = y + self.bias
            
        return y

    def get_log_prior(self):
        '''
        return:
            log_prior = log[HalfCauchy(rank_param | eta)] + log[Normal(factor_1 | 0, rank_param)]
                    + log[Normal(factor_2 | 0, rank_param)] + ... + log[Normal(factor_{M+1} | 0, rank_param)]
        '''
        
        return self.weight_tensor._get_log_prior()