
from tensor_fusion.subnet import *
from tensor_fusion.module import TensorFusion, AdaptiveRankFusion, LowRankFusion
from tensor_fusion.util import *

class TFN(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, out_size, dropouts, device=None, dtype=None):

        super().__init__()

        self.text_subnet = TextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                      device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        self.fusion_layer = TensorFusion(fusion_input_sizes, fusion_size, dropouts[3], device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)

    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output

class LMF(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, rank, out_size, dropouts, device=None, dtype=None):

        super().__init__()

        self.text_subnet = TextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                      device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        self.fusion_layer = LowRankFusion(fusion_input_sizes, fusion_size, rank, dropouts[3], device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)
    
    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output

class AdaptiveRankFusionNetwork(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, max_rank, out_size, dropouts, 
                 tensor_type='CP', prior_type='log_uniform', fusion_max_rank=20, eta=None, device=None, dtype=None):

        super().__init__()

        self.text_subnet = AdaptiveRankTextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropouts[0],
                                                  max_rank=max_rank, tensor_type=tensor_type, prior_type=prior_type, eta=eta,
                                                  device=device, dtype=dtype)

        self.audio_subnet = AdaptiveRankSubNet(input_sizes[1], hidden_sizes[1], dropouts[1],
                                               max_rank=max_rank, tensor_type=tensor_type, prior_type=prior_type, eta=eta,
                                               device=device, dtype=dtype)

        self.video_subnet = AdaptiveRankSubNet(input_sizes[2], hidden_sizes[2], dropouts[2],
                                               max_rank=max_rank, tensor_type=tensor_type, prior_type=prior_type, eta=eta,
                                               device=device, dtype=dtype)

        fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        self.fusion_layer = AdaptiveRankFusion(fusion_input_sizes, fusion_size, dropouts[3],
                                               max_rank=fusion_max_rank, prior_type=prior_type, eta=eta,
                                               device=device, dtype=dtype)

        self.inference_subnet = AdaptiveRankInferenceSubNet(fusion_size, out_size, dropouts[4],
                                                            max_rank=max_rank, tensor_type=tensor_type, prior_type=prior_type, eta=eta,
                                                            device=device, dtype=dtype)


    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output

    def get_log_prior(self):

        return self.text_subnet.get_log_prior() + self.audio_subnet.get_log_prior() + self.video_subnet.get_log_prior() + \
            self.fusion_layer.get_log_prior() + self.inference_subnet.get_log_prior()