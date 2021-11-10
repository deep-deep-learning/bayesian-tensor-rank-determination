import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np

from tensor_layers import low_rank_tensors

from model import SubNet, TextSubNet

class CP_linear(nn.Linear):
    '''
    Implements CP layer with factorized forward propagation

    y = ((x_1 @ W_1) * (x_2 @ W_2) * ... * (x_M @ W_M)) @ W_y.T

    where
        x_m is an input matrix with the shape of (batch_size, x_m_size) for m = 1, 2, ..., M

        W_m is a CP-decomposition factor matrix with the shape of (x_m_size, max_rank) for m = 1, 2, ..., M

        W_y is a CP-decomposition factor matrix with the shape of (output_size, max_rank)
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=20,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta=None,
                 device=None,
                 dtype=None,
                 ):

        super(CP_linear, self).__init__(in_features, out_features, bias, device, dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2 / self.in_features)

        # weight tensor (CP)
        self.tensor = getattr(low_rank_tensors, tensor_type)(shape, prior_type=prior_type, em_stepsize=em_stepsize,
                                                             max_rank=max_rank, initialization_method='nn',
                                                             target_stddev=target_stddev, learned_scale=False, eta=eta)

    def forward(self, inputs, rank_update=True):
        '''
        Performs y = ((x_1 @ W_1) * (x_2 @ W_2) * ... * (x_M @ W_M)) @ W_y.T and rank update

        Args:
            inputs - a length-M list that contains [x_1, x_2, ..., x_M]
                        where x_m is a matrix with the shape of (batch_size, x_1_size)
            rank_update - a boolean flag for rank update

        Returns:
            y - a matrix with shape (batch_size, output_size)
        '''

        # update rank parameters
        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        # initialize y = 1
        y = torch.ones(size=(1,))

        for i, x in enumerate(inputs):
            # y *= (x_m @ W_m)
            y = y * (x @ self.tensor.factors[i])

        y = y @ self.tensor.factors[i + 1].T

        return y

    def update_rank_parameters(self):

        self.tensor.update_rank_parameters()


class CP_tensor_fusion_network(nn.Module):
    '''

    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    with rank-adaptive tensorized training from Hawkins, Cole and Zheng Zhang "Bayesian tensorized
    neural networks with automatic rank selection." Neurocomputing 2021.

    '''

    def __init__(self, input_sizes, hidden_sizes, output_size, max_rank):
        '''
        Args:
            input_sizes - a length-3 tuple that contains (x_1_size, x_2_size, x_3_size)
            hidden_sizes - a length-3 tuple that contains (hidden_size_1, hidden_size_2, hidden_size_3)
            output_size - an integer specifying the size of the output
            max_rank - an integer specifying the maximum rank of weight tensor
        '''

        super(CP_tensor_fusion_network, self).__init__()

        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.max_rank = max_rank

        # subnets for audio, video, and text inputs
        # they each output z_audio, z_video, z_text
        self.audio_subnet = SubNet(input_sizes[0], hidden_sizes[0], dropout=0.3)
        self.video_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropout=0.3)
        self.text_subnet = TextSubNet(input_sizes[2], hidden_sizes[2], hidden_sizes[2], dropout=0.3)

        # rank adaptive tensor layer for y = ((z_audio @ W_audio) * (z_video @ W_video) * (x_text @ W_text)) @ W_y.T
        # the shape of weight tensor is (hidden_size_audio + 1, hidden_size_video + 1, hidden_size_text + 1, output_size)
        shape = (hidden_sizes[0] + 1, hidden_sizes[1] + 1, hidden_sizes[2] + 1, output_size)

        self.tensor_fusion_layer = CP_linear(in_features=np.prod(hidden_sizes), out_features=output_size,
                                             shape=shape, tensor_type='CP', max_rank=max_rank)

    def forward(self, inputs):
        '''
        Performs:
            z_audio = (Subnet(x_audio), 1)
            z_video (Subnet(x_video), 1)
            z_text = (TextSubnet(x_text), 1)
            output = TensorFusionLayer([z_audio, z_video, z_text])

        Args:
            inputs - a length-M list that contains (x_audio, x_video, x_text)

        Returns:
            output - an output matrix with the shape of (batch_size, output_size)
        '''

        # subnet outputs
        z_audio = self.audio_subnet(inputs[0])
        z_video = self.video_subnet(inputs[1])
        z_text = self.text_subnet(inputs[2])

        batch_size = z_audio.data.shape[0]

        if z_audio.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # 1 in concatenated to each subnet outputs
        z_audio = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), z_audio), dim=1)
        z_video = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), z_video), dim=1)
        z_text = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), z_text), dim=1)

        output = self.tensor_fusion_layer([z_audio, z_video, z_text])

        return output