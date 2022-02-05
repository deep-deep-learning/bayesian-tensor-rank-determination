import torch
import torch.nn as nn
from tltorch.factorized_layers.factorized_linear import FactorizedLinear
from tltorch.factorized_tensors import TensorizedTensor

class FactorizedLinearCP(FactorizedLinear):
    
    def __init__(self, in_tensorized_features, out_tensorized_features, bias=False,
                 rank=10, device=None, dtype=None):
        '''
        args:
            in_tensorized_features: a tuple of ints, (in_size_1, in_size_2, ..., in_size_n) 
            out_tensorized_features: a tuple of ints, (out_size_1, out_size_2, ..., out_size_m)
            bias: a boolean, True for bias False for no bias
            max_rank: maximum rank for CP decomposition of weight
        '''
        
        super(FactorizedLinearCP, self).__init__(in_tensorized_features, out_tensorized_features, bias,
                                       factorization='CP', rank=rank, n_layers=1, 
                                       device=device, dtype=dtype)

        self.n_input_factors = len(in_tensorized_features)
        self.n_output_factors = len(out_tensorized_features)
    
    def from_matrix(self, matrix, bias=None):
        '''
        changes the weight and bias as given
        '''

        self.weight = TensorizedTensor.from_matrix(matrix, 
                                                   self.out_tensorized_features, 
                                                   self.in_tensorized_features, 
                                                   self.rank, 
                                                   factorization='CP')
        if bias is None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(bias)       
        
    def forward(self, x):
        '''
        X @ W.T + b
        
        factors are in the order of [out_factors, in_factors]
        '''
        
        # tensorize input
        output = x.reshape((x.shape[0],) + self.in_tensorized_features)
        
        # forward propagate with input factors
        output = torch.einsum('na...,ar->n...r', output, self.weight.factors[self.n_output_factors])
        for factor in self.weight.factors[self.n_output_factors+1:]:
            output = torch.einsum('na...r,ar->n...r', output, factor)
            
        # forward propagate with output factors
        for factor in self.weight.factors[:self.n_output_factors-1]:
            output = torch.einsum('n...r,ar->n...ar', output, factor)
        output = torch.einsum('n...r,ar->n...a', output, self.weight.factors[self.n_output_factors-1])
        
        # vectorize output
        output = output.reshape((x.shape[0], self.out_features))
        
        # add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output