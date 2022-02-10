import torch
import torch.nn as nn
import tltorch
from tensor_fusion.fusion_layer import CP_with_trainable_rank_parameter
import numpy as np
from tensorized_fwd_bwd.tensor_times_matrix import tensor_times_matrix_fwd

class AdaptiveRankFactorizedLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            in_features: input dimension size
            out_features: output dimension size
            dropout: dropout probability
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
        '''
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        tensorized_shape = tltorch.utils.get_tensorized_shape(in_features, out_features, verbose=False)
        dims = tensorized_shape[0] + tensorized_shape[1]
        target_stddev = np.sqrt(2/in_features)
        if tensor_type == 'CP':
            self.weight_tensor = CP_with_trainable_rank_parameter(dims,
                                                                  prior_type=prior_type, 
                                                                  max_rank=max_rank,
                                                                  tensorized_shape=tensorized_shape,
                                                                  initialization_method='nn',
                                                                  target_stddev=target_stddev,
                                                                  learned_scale=False,
                                                                  eta=eta,
                                                                  device=device,
                                                                  dtype=dtype)

        self.weight_tensor.to(dtype)
        self.weight_tensor.to(device)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), device=device, dtype=dtype))
        else:
            self.bias = None
            
    def forward(self, x):
        
        output = tensor_times_matrix_fwd(self.weight_tensor, x.T)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output