import torch
from torch.distributions.half_cauchy import HalfCauchy
import torch.distributions as td
from tensor_layers.low_rank_tensors import CP, TensorTrain
import numpy as np

class CP_with_trainable_rank_parameter(CP):
    
    def __init__(self, dims, max_rank, learned_scale=True, tensorized_shape=None, **kwargs):
        
        super().__init__(dims, max_rank, learned_scale, **kwargs)

        self.rank_parameter_distribution = HalfCauchy(self.eta)
        if tensorized_shape is not None:
            self.tensorized_shape = tensorized_shape
            self.shape = (np.prod(tensorized_shape[0]), np.prod(tensorized_shape[1]))
            self.name = self.tensor_type
        
        
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