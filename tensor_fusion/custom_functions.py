import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error

from model import SubNet, TextSubNet, TFN

import numpy as np

class CP_Linear_Function_3(torch.autograd.Function):
    '''
    
    Implements the torch.autograd.Function for a linear layer with its weight tensor W, 
    CP decomposed into factor_1, factor_2, factor_3, and factor_4 for a given rank 'rank' and
    an input tensor X that can be CP decomposed into vectors x_1, x_2, and x_3 (i.e., 
    the rank is 1).
    
    '''
    @staticmethod
    def forward(ctx, x_1, x_2, x_3, factor_1, factor_2, factor_3, factor_y):
        '''
        
        Implements the forward propagation that takes advantage of tensor compuatation 
        
        Args:
            x_1: input for the first mode - (batch_size, input_size_1)
            x_2: input for the second mode - (batch_size, input_size_2)
            x_3: input for the third mode - (batch_size, input_size_3)
            
            factor_1: factors for the first mode of W tensor - (input_size_1, 'rank')
            factor_2: factors for the second mode of W tensor - (input_size_2, 'rank')
            factor_3: factors for the third mode of W tensor - (input_size_3, 'rank')
            factor_4: factors for the fourth mode of W tensor - (input_size_4, 'rank')
            
        Returns:
            y: output for linear operation XW where W is the weight tensor and X is the input
               tensor
        
        '''
        # y = XW can be simplified into 
        # y = {(x_1 @ factor_1) (x_2 @ factor_2) (x_3 @ factor_3)} @ factor_4.T
        
        A_1 = x_1 @ factor_1
        A_2 = x_2 @ factor_2
        A_3 = x_3 @ factor_3
        
        A_f = A_1 * A_2 * A_3
        
        y = A_f @ factor_4.T
        
        # saved for barckward propagation
        ctx.save_for_backward(x_1, x_2, x_3, factor_1, factor_2, factor_3, factor_4, 
                              A_1, A_2, A_3, A_f)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        '''
        
        Implements the backward propagation that takes advantage of tensor computation
        
        Args:
            grad_y: the gradient of y with respect to the loss
            
        Returns:
            grad_x_1: the gradient of x_1 w.r.t. the loss
            grad_x_2: the gradient of x_2 w.r.t. the loss
            grad_x_3: the gradient of x_3 w.r.t. the loss
            grad_factor_1: the gradient of factor_1 w.r.t. the loss
            grad_factor_2: the gradient of factor_2 w.r.t. the loss
            grad_factor_3: the gradient of factor_3 w.r.t. the loss
            grad_factor_4: the gradient of factor_4 w.r.t. the loss
        
        '''
        x_1, x_2, x_3, factor_1, factor_2, factor_3, factor_4, \ 
        A_1, A_2, A_3, A_f = ctx.saved_tensors
        
        grad_x_1 = grad_x_2 = grad_x_3 = grad_factor_1 = grad_factor_2 = \
        grad_factor_3 = grad_factor_4 = None
        
        
        grad_factor_4 = grad_y.T @ A_f
            
        grad_A_f = grad_y @ factor_4
        
        grad_A_1 = grad_A_f * A_2 * A_3
        grad_A_2 = grad_A_f * A_1 * A_3
        grad_A_3 = grad_A_f * A_1 * A_2
        
        grad_factor_1 = x_1.T @ grad_A_1
        grad_factor_2 = x_2.T @ grad_A_2
        grad_factor_3 = x_3.T @ grad_A_3
        
        grad_x_1 = grad_A_1 @ factor_1.T
        grad_x_2 = grad_A_2 @ factor_2.T
        grad_x_3 = grad_A_3 @ factor_3.T
        
        return grad_x_1, grad_x_2, grad_x_3, grad_W_1, grad_W_2, grad_W_3, grad_W_y