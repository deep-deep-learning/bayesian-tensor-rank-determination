import torch
import torch.nn as nn

import numpy as np

from tensor_layers import low_rank_tensors

class Adaptive_Rank_CP_Linear(nn.Module):

    def __init__(self, input_sizes, output_size, max_rank=20, em_stepsize=1.0, 
                 prior_type='log_uniform', eta=None):

        super(Adaptive_Rank_CP_Linear, self).__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        
        shape = input_sizes + (output_size,)
        target_stddev = np.sqrt(2 / np.prod(self.input_sizes))
        self.weight_tensor = getattr(low_rank_tensors, 'CP')(shape, prior_type=prior_type, em_stepsize=em_stepsize,
                                                      max_rank=max_rank, initialization_method='nn', 
                                                      target_stddev=target_stddev, learned_scale=False, 
                                                      eta=eta)

    def forward(self, inputs, rank_update=True):
        
        if self.training and rank_update:
            self.weight_tensor.update_rank_parameters()
        
        y = torch.ones(size=(1,))
        for i, x in enumerate(inputs):
            y = y * (x @ self.weight_tensor.factors[i])
        y = y @ self.weight_tensor.factors[-1].T

        return y

class Fixed_Rank_CP_Linear(nn.Module):
    
    def __init__(self, input_sizes, output_size, rank=20):
        
        super(Fixed_Rank_CP_Linear, self).__init__()
        
        self.input_sizes = input_sizes
        self.output_size = output_size
        self.rank = rank
        
        self.weight_tensor_factors = self.initialize_weight_tensor_factors()
        
    def forward(self, inputs):
        
        # y = ((x_1 @ W_1)(x_2 @ W_2)...(x_M @ W_M)) @ W_y.T
        y = torch.ones(size=(1,))
        for i, x in enumerate(inputs):
            y = y * (x @ self.weight_tensor_factors[i])
        y = y @ self.weight_tensor_factors[-1].T

        return y
        
    def initialize_weight_tensor_factors(self):
        
        factors = []
        for m, input_size in enumerate(self.input_sizes):
            factors.append(nn.Parameter(torch.empty(input_size, self.rank)))
            nn.init.xavier_normal_(factors[m])
        factors.append(nn.Parameter(torch.empty(self.output_size, self.rank)))
        nn.init.xavier_normal_(factors[-1])
            
        return nn.ParameterList(factors)