import torch
import torch.nn as nn
from .fusion_layer import AdaptiveRankFusionLayer
from .model import AdaptiveRankTensorizedTextSubNet, AdaptiveRankTensorizedSubNet
from .other_model import SubNet, TextSubNet

class AdaptiveRankFusion_with_AdaptiveRankSubNets(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, dropouts, output_size, bias=False,
                 max_rank=10, prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            input_sizes: a tuple of ints, (audio_in, video_in, ... text_in)
            hidden_sizes: a tuple of ints, (audio_hidden, video_hidden, ... text_hidden)
            dropouts: a tuple of floats, (dropout_1, dropout_2, ..., dropout_M, post_fusion_dropout)
            output_size: an int, output size for fusion layer
            max_rank: an int, maximum rank for the CP decomposition
        '''
        super().__init__()
        
        # define the pre-fusion subnetworks
        self.audio_subnet = AdaptiveRankTensorizedSubNet(input_sizes[0], hidden_sizes[0], dropouts[0], bias=bias,
                                                         max_rank=max_rank, prior_type=prior_type, eta=eta,
                                                         device=device, dtype=dtype)
        self.video_subnet = AdaptiveRankTensorizedSubNet(input_sizes[1], hidden_sizes[1], dropouts[1], bias=bias,
                                                         max_rank=max_rank, prior_type=prior_type, eta=eta,
                                                         device=device, dtype=dtype)
        self.text_subnet = AdaptiveRankTensorizedTextSubNet(input_sizes[2], hidden_sizes[2], hidden_sizes[2]//2, dropout=dropouts[2], bias=bias,
                                                            max_rank=max_rank, prior_type=prior_type, eta=eta,
                                                            device=device, dtype=dtype)
        
        fusion_input_sizes = (hidden_sizes[0]+1, hidden_sizes[1]+1, hidden_sizes[2]//2+1)
        # define fusion layer
        self.fusion_layer = AdaptiveRankFusionLayer(input_sizes=fusion_input_sizes,
                                                    output_size=output_size,
                                                    max_rank=max_rank,
                                                    prior_type=prior_type,
                                                    eta=eta,
                                                    device=device,
                                                    dtype=dtype)

        self.post_fusion_dropout = nn.Dropout(dropouts[-1])

    def forward(self, audio_x, video_x, text_x):

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        batch_size = audio_h.shape[0]
        device = audio_h.device
        dtype = audio_h.dtype

        audio_h = torch.cat((audio_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)
        video_h = torch.cat((video_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)
        text_h = torch.cat((text_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)

        output = self.fusion_layer([audio_h, video_h, text_h])
        output = self.post_fusion_dropout(output)
        
        return output

    def get_log_prior(self):

        return self.audio_subnet.get_log_prior() + self.text_subnet.get_log_prior() + self.video_subnet.get_log_prior() + \
            self.fusion_layer.get_log_prior()

class AdaptiveRankFusion(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, dropouts, output_size, max_rank=10, 
                 prior_type='half_cauchy', eta=None,
                 device=None, dtype=None):
        '''
        args:
            input_sizes: a tuple of ints, (audio_in, video_in, ... text_in)
            hidden_sizes: a tuple of ints, (audio_hidden, video_hidden, ... text_hidden)
            dropouts: a tuple of floats, (dropout_1, dropout_2, ..., dropout_M, post_fusion_dropout)
            output_size: an int, output size for fusion layer
            max_rank: an int, maximum rank for the CP decomposition
        '''
        super(AdaptiveRankFusion, self).__init__()
        
        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(input_sizes[0], hidden_sizes[0], dropouts[0], device=device, dtype=dtype)
        self.video_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)
        self.text_subnet = TextSubNet(input_sizes[2], hidden_sizes[2], hidden_sizes[2]//2, dropout=dropouts[2], device=device, dtype=dtype)
        
        fusion_input_sizes = (hidden_sizes[0]+1, hidden_sizes[1]+1, hidden_sizes[2]//2+1)
        # define fusion layer
        self.fusion_layer = AdaptiveRankFusionLayer(input_sizes=fusion_input_sizes,
                                                    output_size=output_size,
                                                    max_rank=max_rank,
                                                    prior_type=prior_type,
                                                    eta=eta,
                                                    device=device,
                                                    dtype=dtype)

        self.post_fusion_dropout = nn.Dropout(dropouts[-1])

    def forward(self, audio_x, video_x, text_x):

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        batch_size = audio_h.shape[0]
        device = audio_h.device
        dtype = audio_h.dtype

        audio_h = torch.cat((audio_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)
        video_h = torch.cat((video_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)
        text_h = torch.cat((text_h, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1)

        output = self.fusion_layer([audio_h, video_h, text_h])
        output = self.post_fusion_dropout(output)
        
        return output
    
    def get_log_prior(self):

        return self.fusion_layer.get_log_prior()