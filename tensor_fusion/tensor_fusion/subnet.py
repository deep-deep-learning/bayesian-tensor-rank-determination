import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import AdaptiveRankLinear, AdaptiveRankLSTM

class InferenceSubNet(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0, device=None, dtype=None):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, in_size, device=device, dtype=dtype)
        self.linear_2 = nn.Linear(in_size, out_size, device=device, dtype=dtype)

    def forward(self, x):
        output = F.relu(self.linear_1(x))
        output = self.dropout(output)
        output = self.linear_2(output)
        return output

class AdaptiveRankInferenceSubNet(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0, bias=True,
                 max_rank=20, tensor_type='CP', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.linear_1 = AdaptiveRankLinear(in_size, in_size, bias=bias,
                                           max_rank=max_rank, tensor_type=tensor_type,
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)

        self.linear_2 = AdaptiveRankLinear(in_size, out_size, bias=bias,
                                           max_rank=max_rank,tensor_type=tensor_type,
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)

    def forward(self, x):


        output = F.relu(self.linear_1(x))
        output = self.dropout(output)
        output = self.linear_2(output)
        return output

    def get_log_prior(self):

        return self.linear_1.weight_tensor._get_log_prior() + self.linear_2.weight_tensor._get_log_prior()

class SubNet(nn.Module):
    '''
    From https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
    '''
    def __init__(self, in_size, hidden_size, dropout=0.2, device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size, device=device, dtype=dtype)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size, device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)

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

class AdaptiveRankSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, dropout=0.2, bias=True,
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
        
        self.dropout = nn.Dropout(p=dropout)

        self.linear_1 = AdaptiveRankLinear(in_size, hidden_size, bias=bias,
                                           max_rank=max_rank, tensor_type=tensor_type,
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)

        self.linear_2 = AdaptiveRankLinear(hidden_size, hidden_size, bias=bias,
                                           max_rank=max_rank,tensor_type=tensor_type,
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)

        self.linear_3 = AdaptiveRankLinear(hidden_size, hidden_size, bias=bias,
                                           max_rank=max_rank, tensor_type=tensor_type,
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)       

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.dropout(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3
    
    def get_log_prior(self):

        return self.linear_1.weight_tensor._get_log_prior() + \
                self.linear_2.weight_tensor._get_log_prior() + \
                    self.linear_3.weight_tensor._get_log_prior()

class TextSubNet(nn.Module):
    '''
    From https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
    '''
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size, device=device, dtype=dtype)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = torch.sigmoid(self.linear_1(h))
        return y_1

class AdaptiveRankTextSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2, bias=True,
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

        self.rnn = AdaptiveRankLSTM(in_size, hidden_size, bias,
                                    max_rank=max_rank, tensor_type=tensor_type,
                                    prior_type=prior_type, eta=eta,
                                    device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
        
        self.linear_1 = AdaptiveRankLinear(hidden_size, out_size, bias,
                                           max_rank=max_rank, tensor_type=tensor_type, 
                                           prior_type=prior_type, eta=eta,
                                           device=device, dtype=dtype)
        

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = torch.sigmoid(self.linear_1(h))

        return y_1

    def get_log_prior(self):

        return self.rnn.get_log_prior() + self.linear_1.get_log_prior()