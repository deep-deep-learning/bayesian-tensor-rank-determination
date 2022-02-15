from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
from tensor_fusion.layer import AdaptiveRankTensorizedLinear

class AdaptiveRankTensorizedSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, dropout, bias=False,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
            max_rank: maximum rank for the weight tensors
            tensor_type: weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size, device=device, dtype=dtype)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = AdaptiveRankTensorizedLinear(in_size, hidden_size, bias=bias,
                                                     max_rank=max_rank,
                                                     tensor_type=tensor_type,
                                                     prior_type=prior_type,
                                                     eta=eta,
                                                     device=device,
                                                     dtype=dtype)
        self.linear_2 = AdaptiveRankTensorizedLinear(hidden_size, hidden_size, bias=bias,
                                                     max_rank=max_rank,
                                                     tensor_type=tensor_type,
                                                     prior_type=prior_type,
                                                     eta=eta,
                                                     device=device,
                                                     dtype=dtype)
        self.linear_3 = AdaptiveRankTensorizedLinear(hidden_size, hidden_size, bias=bias,
                                                     max_rank=max_rank,
                                                     tensor_type=tensor_type,
                                                     prior_type=prior_type,
                                                     eta=eta,
                                                     device=device,
                                                     dtype=dtype)       

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3
    
    def get_log_prior(self):

        return self.linear_1.weight_tensor._get_log_prior() + \
                self.linear_2.weight_tensor._get_log_prior() + \
                    self.linear_3.weight_tensor._get_log_prior()

class AdaptiveRankTensorizedLSTM(nn.Module):
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
        
        self.layer_ih = AdaptiveRankTensorizedLinear(input_size, hidden_size*4, bias, 
                                                     max_rank=max_rank, 
                                                     tensor_type=tensor_type, 
                                                     prior_type=prior_type, 
                                                     eta=eta,
                                                     device=device, 
                                                     dtype=dtype)
        self.layer_hh = AdaptiveRankTensorizedLinear(hidden_size, hidden_size*4, bias,
                                                     max_rank=max_rank, 
                                                     tensor_type=tensor_type, 
                                                     prior_type=prior_type, 
                                                     eta=eta,
                                                     device=device,
                                                     dtype=dtype)
        
    def forward(self, x):
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

class AdaptiveRankTensorizedTextSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2, bias=False,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            out_size: output dimension
            dropout: dropout probability
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
            
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.rnn = AdaptiveRankTensorizedLSTM(in_size, hidden_size, bias,
                                              max_rank=max_rank, tensor_type=tensor_type,
                                              prior_type=prior_type, eta=eta,
                                              device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size, device=device, dtype=dtype)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

    def get_log_prior(self):

        return self.rnn.get_log_prior()