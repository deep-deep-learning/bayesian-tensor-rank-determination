import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal
import numpy as np
from .distribution import LogUniform
from . import low_rank_tensor as LowRankTensor
from .quantized_tensor_times_matrix import tensor_times_matrix_fwd

class QuantizedAdaptiveRankFusion(nn.Module):

    def __init__(self, quantizer, in_features, out_features, tensorized_shape, bias=True, 
                 max_rank=20, prior_type='log_uniform', eta=None, 
                 device=None, dtype=None):

        super().__init__()

        self.quantizer = quantizer

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight_tensor = getattr(LowRankTensor, 'CP')(in_features, out_features, max_rank, prior_type, eta, device, dtype, tensorized_shape)
        
       # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), 0.1, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):

        # tensorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor.tensor.factors[:-1]):
            output = output * (x @ self.quantizer(factor))

        output = output @ self.quantizer(self.weight_tensor.tensor.factors[-1].T)

        if self.bias is not None:
            output = self.quantizer(output + self.bias)

        output = F.relu(output)
            
        return output

    def get_log_prior(self):

        return self.weight_tensor.get_log_prior()

    def estimate_rank(self):

        return self.weight_tensor.estimate_rank()

class QuantizedAdaptiveRankLinear(nn.Module):

    def __init__(self, quantizer, in_features, out_features, bias=True, 
                 max_rank=20, tensor_type='TT', prior_type='log_uniform',
                 eta=None, device=None, dtype=None):

        super().__init__()

        self.quantizer = quantizer

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight_tensor = getattr(LowRankTensor, tensor_type)(in_features, out_features, max_rank, prior_type, eta, device, dtype)
        
       # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), 0.1, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):

        output = tensor_times_matrix_fwd(self.weight_tensor.tensor, x.T, self.quantizer)

        if self.bias is not None:
            output = output + self.bias
            
        return output

    def get_log_prior(self):

        return self.weight_tensor.get_log_prior()

    def estimate_rank(self):

        return self.weight_tensor.estimate_rank()