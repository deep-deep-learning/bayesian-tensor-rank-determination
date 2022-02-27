import torch
import torch.nn as nn
from torch.distributions.half_cauchy import HalfCauchy
import torch.distributions as td
from tensor_layers.low_rank_tensors import CP, TensorTrain, Tucker, TensorTrainMatrix
import numpy as np
from .distribution import LogUniform
import tensorly as tl
tl.set_backend('pytorch')

class CP_with_trainable_rank_parameter(CP):
    
    def __init__(self, dims, max_rank, learned_scale=True, tensorized_shape=None, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        if self.prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(self.eta)
        elif self.prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=self.device, dtype=self.dtype), 
                                                                torch.tensor([1e30], device=self.device, dtype=self.dtype))

        if tensorized_shape is not None:
            self.tensorized_shape = tensorized_shape
            self.shape = (np.prod(tensorized_shape[0]), np.prod(tensorized_shape[1]))
            self.name = self.tensor_type

        self.threshold = nn.Threshold(1e-30, 1e-30, inplace=True)
        
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
        clamped_rank_parameter = self.rank_parameter.clamp(1e-30)
        self.rank_parameter.data = clamped_rank_parameter.data
        
        
        # self.threshold(self.rank_parameter)
        log_prior = torch.sum(self.rank_parameter_prior_distribution.log_prob(self.rank_parameter))
        
        # 0 mean normal distribution for the factors
        for factor_prior_distribution, factor in zip(self.factor_prior_distributions, 
                                                     self.factors):
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))
        
        return log_prior

class TT_with_trainable_rank_parameter(TensorTrain):
    
    def __init__(self, dims, max_rank, learned_scale=True, tensorized_shape=None, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        if self.prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(self.eta)
        elif self.prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=self.device, dtype=self.dtype), 
                                                                torch.tensor([1e30], device=self.device, dtype=self.dtype))

        self.shape = dims
        self.name = 'TT'
        self.rank = self.max_ranks
    
    def _build_low_rank_prior(self):

        self.rank_parameters = [
            self.add_variable(torch.sqrt(x.clone().detach()).to(self.device),trainable=True)
            for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = []

        for i in range(len(self.dims) - 1):

            self.factor_prior_distributions.append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[i].shape, device=self.device, dtype=self.dtype),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=3))

        self.factor_prior_distributions.append(
            td.Independent(td.Normal(
                loc=torch.zeros(self.factors[-1].shape, device=self.device, dtype=self.dtype),
                scale=self.rank_parameters[-1].unsqueeze(1).unsqueeze(2)),
                            reinterpreted_batch_ndims=3))
    
    def _get_log_prior(self):

        log_prior = 0.0
        for rank_parameter in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            clamped_rank_parameter = rank_parameter.clamp(1e-5)
            rank_parameter.data = clamped_rank_parameter.data
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(rank_parameter))
    
        # 0 mean normal distribution for the factors
        for factor_prior_distribution, factor in zip(self.factor_prior_distributions, 
                                                     self.factors):
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))
        
        return log_prior

class Tucker_with_trainable_rank_parameter(Tucker):
    
    def __init__(self, dims, max_rank, learned_scale=True, tensorized_shape=None, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        if self.prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(self.eta)
        elif self.prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=self.device, dtype=self.dtype), 
                                                                torch.tensor([1e30], device=self.device, dtype=self.dtype))

        if tensorized_shape is not None:
            self.tensorized_shape = tensorized_shape
            self.shape = (np.prod(tensorized_shape[0]), np.prod(tensorized_shape[1]))
            self.name = self.tensor_type
        
        self.core = self.factors[0]
        self.factors = self.factors[1]
        self.name = 'Tucker'
    
    def get_full(self):

        factors = [self.core, self.factors]

        return tl.tucker_to_tensor(factors)

    def _build_low_rank_prior(self, core_prior=10.0):

        self.rank_parameters = [
            self.add_variable(torch.sqrt(x.clone().detach()).to(self.device),trainable=False) for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = (td.Independent(
            td.Normal(loc=torch.zeros(self.factors[0].shape, device=self.device, dtype=self.dtype), scale=core_prior),
            reinterpreted_batch_ndims=len(self.dims)), [])

        for i in range(len(self.dims)):

            self.factor_prior_distributions[1].append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[1][i].shape, device=self.device, dtype=self.dtype),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=2))
    
    def _get_log_prior(self):

        log_prior = 0.0
        for rank_parameter in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            clamped_rank_parameter = rank_parameter.clamp(1e-5)
            rank_parameter.data = clamped_rank_parameter.data
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(rank_parameter))

        # for the core factors
        log_prior = log_prior + torch.sum(self.factor_prior_distributions[0].log_prob(self.core))
        # 0 mean normal distribution for the factors
        for factor_prior_distribution, factor in zip(self.factor_prior_distributions[1], 
                                                     self.factors):
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))
        
        return log_prior

class TTM_with_trainable_rank_parameter(TensorTrainMatrix):
    
    def __init__(self, dims, max_rank, learned_scale=True, tensorized_shape=None, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        if self.prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(self.eta)
        elif self.prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=self.device, dtype=self.dtype), 
                                                                torch.tensor([1e30], device=self.device, dtype=self.dtype))

        self.tensorized_shape = (self.dims1, self.dims2)
        self.name = 'BlockTT'
        self.rank = self.max_ranks

    def _build_low_rank_prior(self):

        self.rank_parameters = [
            self.add_variable(x.clone().detach().to(self.device),trainable=True)
            for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = []

        for i in range(self.num_cores - 1):

            self.factor_prior_distributions.append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[i].shape, device=self.device, dtype=self.dtype),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=4))

        self.factor_prior_distributions.append(
            td.Independent(td.Normal(
                loc=torch.zeros(self.factors[-1].shape, device=self.device, dtype=self.dtype),
                scale=self.rank_parameters[-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)),
                            reinterpreted_batch_ndims=4))
    
    def _get_log_prior(self):

        log_prior = 0.0
        for rank_parameter in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            clamped_rank_parameter = rank_parameter.clamp(1e-5)
            rank_parameter.data = clamped_rank_parameter.data
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(rank_parameter))
    
        # 0 mean normal distribution for the factors
        for factor_prior_distribution, factor in zip(self.factor_prior_distributions, 
                                                     self.factors):
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))

        return log_prior