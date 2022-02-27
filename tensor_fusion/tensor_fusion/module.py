import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import numpy as np


from .low_rank_tensor import CP_with_trainable_rank_parameter, TT_with_trainable_rank_parameter, \
    Tucker_with_trainable_rank_parameter, TTM_with_trainable_rank_parameter

from .tensor_times_matrix import tensor_times_matrix_fwd

class TensorFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        
        # initialize weight tensor
        tensorized_shape = input_sizes + (output_size,)
        self.weight_tensor = nn.Parameter(torch.empty(tensorized_shape, device=device, dtype=dtype))
        nn.init.xavier_normal_(self.weight_tensor)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):

        fusion_tensor = inputs[0]
        for x in inputs[1:]:
            fusion_tensor = torch.einsum('n...,na->n...a', fusion_tensor, x)
        
        fusion_tensor = self.dropout(fusion_tensor)

        output = torch.einsum('n...,...o->no', fusion_tensor, self.weight_tensor)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output)

class LowRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, rank, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor factors
        factors = [nn.Parameter(torch.empty((input_size, rank), device=device, dtype=dtype)) \
            for input_size in input_sizes]
        factors = factors + [nn.Parameter(torch.empty((output_size, rank), device=device, dtype=dtype))]
        
        for factor in factors:
            nn.init.xavier_normal_(factor)

        self.weight_tensor_factors = nn.ParameterList(factors)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):
        
        # tensorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor_factors[:-1]):
            output = output * (x @ factor)
        
        output = self.dropout(output)

        output = output @ self.weight_tensor_factors[-1].T

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output)

class AdaptiveRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True,
                 max_rank=20, prior_type='log_uniform', eta=None, 
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
        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor
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

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, inputs):

        output = 1.0
        for i, x in enumerate(inputs):
            output = output * (x @ self.weight_tensor.factors[i])

        output = self.dropout(output)

        output = output @ self.weight_tensor.factors[-1].T

        if self.bias is not None:
            output = output + self.bias
            
        return F.relu(output)

    def get_log_prior(self):

        return self.weight_tensor._get_log_prior()

class AdaptiveRankLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            in_features: input dimension size
            out_features: output dimension size
            max_rank: maximum rank for weight tensor
            tensor_type: weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
        '''
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        tensorized_shape = tltorch.utils.get_tensorized_shape(in_features, out_features, verbose=False)
        
        # TT, Tucker, TTM does not support un even order
        if len(tensorized_shape[0]) == len(tensorized_shape[1]):
            if tensor_type == 'TTM':
                dims = tensorized_shape
            else:
                dims = tensorized_shape[0] + tensorized_shape[1]
        else:
            if tensor_type == 'TTM':
                dims = ((in_features,), (out_features,))
            else:
                dims = (in_features, out_features)

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
        elif tensor_type == 'TT':
            self.weight_tensor = TT_with_trainable_rank_parameter(dims,
                                                                  prior_type=prior_type,
                                                                  max_rank=max_rank,
                                                                  tensorized_shape=None,
                                                                  initialization_method='nn',
                                                                  target_stddev=target_stddev,
                                                                  learned_scale=False,
                                                                  eta=eta,
                                                                  device=device,
                                                                  dtype=dtype)
        elif tensor_type == 'Tucker':
            self.weight_tensor = Tucker_with_trainable_rank_parameter(dims,
                                                                      prior_type=prior_type,
                                                                      max_rank=max_rank,
                                                                      tensorized_shape=tensorized_shape,
                                                                      initialization_method='nn',
                                                                      target_stddev=target_stddev,
                                                                      learned_scale=False,
                                                                      eta=eta,
                                                                      device=device,
                                                                      dtype=dtype)
            print("Do not set Tucker's max rank too high (less than 15)")
        elif tensor_type == 'TTM':
            self.weight_tensor = TTM_with_trainable_rank_parameter(dims,
                                                                   prior_type=prior_type,
                                                                   max_rank=max_rank,
                                                                   tensorized_shape=None,
                                                                   initialization_method='nn',
                                                                   target_stddev=target_stddev,
                                                                   learned_scale=False,
                                                                   eta=eta,
                                                                   device=device,
                                                                   dtype=dtype)
        else:
            print("Do not support the tensor type")

        self.weight_tensor.to(dtype)
        self.weight_tensor.to(device)
        
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), device=device, dtype=dtype))
            target_stddev = np.sqrt(1/in_features)
            nn.init.uniform_(self.bias, -target_stddev, target_stddev)
        else:
            self.bias = None
            
    def forward(self, x):
        
        output = tensor_times_matrix_fwd(self.weight_tensor, x.T)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def get_log_prior(self):

        return self.weight_tensor._get_log_prior()

class AdaptiveRankLSTM(nn.Module):
    '''
    
    no frills batch first LSTM implementation

    '''
    
    def __init__(self, input_size, hidden_size, bias=False,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            input_size: input dimension size
            hidden_size: hidden dimension size
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
        '''

        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.layer_ih = AdaptiveRankLinear(input_size, hidden_size*4, bias=bias, 
                                           max_rank=max_rank, tensor_type=tensor_type, 
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)
        self.layer_hh = AdaptiveRankLinear(hidden_size, hidden_size*4, bias,
                                           max_rank=max_rank, tensor_type=tensor_type, 
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)
        
    def forward(self, x):

        # LSTM forward propagation
        output = []
        batch_size = x.shape[0]
        
        c = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        h = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        for seq in range(20):
            ih = self.layer_ih(x[:,seq,:])
            hh = self.layer_hh(h)
            i, f, g, o = torch.split(ih + hh, self.hidden_size, 1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            output.append(h.unsqueeze(1))
            
        output = torch.cat(output, dim=1)
        
        return output, (h, c)

    def get_log_prior(self):

        return self.layer_ih.get_log_prior() + self.layer_hh.get_log_prior()