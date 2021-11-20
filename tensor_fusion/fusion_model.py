import torch
import torch.nn as nn

from torch.autograd import Variable

from model import SubNet, TextSubNet

from fusion_layers import Adaptive_Rank_CP_Linear, Fixed_Rank_CP_Linear

class CP_Tensor_Fusion_Network(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    with rank-adaptive tensorized training from Hawkins, Cole and Zheng Zhang "Bayesian tensorized
    neural networks with automatic rank selection." Neurocomputing 2021.
    '''

    def __init__(self, input_sizes, hidden_sizes, output_size, max_rank, 
                 rank_adaptive=True):
        '''
        Args:
            input_sizes - a length-3 tuple that contains (x_1_size, x_2_size, x_3_size)
            hidden_sizes - a length-3 tuple that contains (hidden_size_1, hidden_size_2, hidden_size_3)
            output_size - an integer specifying the size of the output
            max_rank - an integer specifying the maximum rank of weight tensor
        '''

        super(CP_Tensor_Fusion_Network, self).__init__()

        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.max_rank = max_rank
        self.rank_adaptive = rank_adaptive
        
        self.audio_subnet = SubNet(input_sizes[0], hidden_sizes[0], dropout=0.3)
        self.video_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropout=0.3)
        self.text_subnet = TextSubNet(input_sizes[2], hidden_sizes[2], hidden_sizes[2], dropout=0.3)
        
        tensor_input_sizes = (hidden_sizes[0] + 1, hidden_sizes[1] + 1, hidden_sizes[2] + 1)
        if rank_adaptive:
            self.tensor_fusion_layer = Adaptive_Rank_CP_Linear(tensor_input_sizes, output_size,
                                                               max_rank=max_rank, em_stepsize=1.0,
                                                               prior_type='log_uniform', eta=None)
        else:
            self.tensor_fusion_layer = Fixed_Rank_CP_Linear(tensor_input_sizes, output_size,
                                                            rank=max_rank)

    def forward(self, inputs):
        
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