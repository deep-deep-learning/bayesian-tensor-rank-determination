import torch
import torch.nn as nn
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal
from tensor_layers.low_rank_tensors import CP
import torch.distributions as td
import numpy as np

class CP_with_trainable_rank_parameter(CP):
    
    def __init__(self, dims, max_rank, learned_scale=True, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        self.rank_parameter_distribution = HalfCauchy(self.eta)
        
        
    def _build_low_rank_prior(self):

        self.rank_parameter = self.add_variable(torch.sqrt(self.get_rank_parameters_update().clone().detach()).view([1,self.max_rank]).to(self.device),
                                                trainable=True)

        self.factor_prior_distributions = []

        for x in self.dims:
            zero_mean = torch.zeros([x, self.max_rank], device=self.device, dtype=self.dtype)
            base_dist = td.Normal(loc=zero_mean,scale=self.rank_parameter)
            independent_dist = td.Independent(base_dist,reinterpreted_batch_ndims=2)
            self.factor_prior_distributions.append(independent_dist)
    
    def _get_log_prior(self):

        # clamp rank_param because <=0 is undefined 
        clamped_rank_parameter = self.rank_parameter.clamp(1e-5)
        self.rank_parameter.data = clamped_rank_parameter.data

        log_prior = torch.sum(self.rank_parameter_distribution.log_prob(self.rank_parameter))
    
        # 0 mean normal distribution for the factors
        for factor_prior_distribution, factor in zip(self.factor_prior_distributions, 
                                                     self.factors):
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))
        
        return log_prior



class AdaptiveRankFusionLayer(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0,
                 max_rank=10, prior_type='half_cauchy', eta=None, 
                 device=None, dtype=None):
        '''
        args:
            input_sizes: a tuple of ints, (input_size_1, input_size_2, ..., input_size_M)
            output_sizes: an int, output size of the fusion layer
            dropout: a float, dropout probablity after fusion
            max_rank: an int, maximum rank for the CP decomposition
            eta: a float, hyperparameter for rank parameter distribution
        '''
        super(AdaptiveRankFusionLayer, self).__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.max_rank = max_rank
        self.eta = eta
        self.device = device
        self.dtype = dtype

        tensorized_shape = input_sizes + (output_size,)
        target_stddev = np.sqrt(2/np.prod(input_sizes))
        self.weight_tensor = CP_with_trainable_rank_parameter(tensorized_shape, 
                                                              prior_type=prior_type, 
                                                              max_rank=max_rank, 
                                                              initialization_method='nn',
                                                              target_stddev=target_stddev,
                                                              learned_scale=False,
                                                              eta=eta,
                                                              device=device,
                                                              dtype=dtype)
        
        self.weight_tensor.to(dtype)
        self.weight_tensor.to(device)
        
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

        return y

    def get_log_prior(self):
        '''
        return:
            log_prior = log[HalfCauchy(rank_param | eta)] + log[Normal(factor_1 | 0, rank_param)]
                    + log[Normal(factor_2 | 0, rank_param)] + ... + log[Normal(factor_{M+1} | 0, rank_param)]
        '''
        
        return self.weight_tensor._get_log_prior()

class AdaptiveRankFusionLayer_1(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, max_rank=10, eta=0.01):
        '''
        args:
            input_sizes: a tuple of ints, (input_size_1, input_size_2, ..., input_size_M)
            output_sizes: an int, output size of the fusion layer
            dropout: a float, dropout probablity after fusion
            max_rank: an int, maximum rank for the CP decomposition
            eta: a float, hyperparameter for rank parameter distribution
        '''
        super(AdaptiveRankFusionLayer, self).__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.max_rank = max_rank
        self.eta = eta

        # CP decomposition factors for the weight tensor
        self.factors = nn.ParameterList([nn.init.xavier_normal_(nn.Parameter(torch.empty(s, max_rank))) 
                                        for s in input_sizes+(output_size,)])
        # rank parameter and its distribution for adaptive rank
        self.rank_param = nn.Parameter(torch.rand((max_rank,)))
        self.rank_param_dist = HalfCauchy(eta)

    def forward(self, inputs):
        '''
        args:
            inputs: a list of vectors, (input_1, input_2, ..., input_M)
        return:
            y = [(input_1 @ factor_1) (input_2 @ factor_2) ... (input_M @ factor_M)] @ factor_{M+1}.T
        '''

        y = 1.0
        for i, x in enumerate(inputs):
            y = y * (x @ self.factors[i])
        y = y @ self.factors[-1].T

        return y

    def get_log_prior(self):
        '''
        return:
            log_prior = log[HalfCauchy(rank_param | eta)] + log[Normal(factor_1 | 0, rank_param)]
                    + log[Normal(factor_2 | 0, rank_param)] + ... + log[Normal(factor_{M+1} | 0, rank_param)]
        '''
        # clamp rank_param because <=0 is undefined 
        clamped_rank_param = self.rank_param.clamp(0.01)
        log_prior = torch.sum(self.rank_param_dist.log_prob(clamped_rank_param))

        # 0 mean normal distribution for the factors
        factor_dist = Normal(0, clamped_rank_param)
        for factor in self.factors:
            log_prior = log_prior + torch.sum(factor_dist.log_prob(factor))
        
        return log_prior


'''
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
'''
