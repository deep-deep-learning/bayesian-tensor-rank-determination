import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

import numpy as np

class MultimodalDataset(Dataset):
    '''
    Dataset for CMU-MOSI
    '''
    def __init__(self, text, audio, vision, labels):
        '''
        args:
            text: text modality feature of shape (N, seq. length, text_input_size)
            audio: audio modality feature of shape (N, seq. length, audio_input_size)
            vision: vision modality feature of shape (N, seq. length, vision_input_size)
            labels: labels of shape (N, 1) and ranges from -3 to 3
        '''
        self.text = text
        self.audio = audio
        self.vision = vision
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns an individual data composed of (features, label)
        where features is a dictionary {'text': , 'audio':, 'vision':}
        Returns:
            features['text']: text modality feature of shape (seq. length, text_input_size)
            features['audio']: audio modality feature of shape (audio_input_size)
            features['vision']: vision modality feature of shape (vision_input_size)
            label: a scalar label that ranges from -3 to 3
        '''
        features = dict()
        features['text'] = self.text[idx]
        # audio and vision features are averaged across time
        features['audio'] = np.mean(self.audio[idx], axis=0)
        features['vision'] = np.mean(self.vision[idx], axis=0)
        label = self.labels[idx]

        return features, label
    
class Multimodal_Binary_Dataset(Dataset):
    '''
    Dataset for CMU-MOSI
    '''
    def __init__(self, text, audio, vision, labels):
        '''
        args:
            text: text modality feature of shape (N, seq. length, text_input_size)
            audio: audio modality feature of shape (N, seq. length, audio_input_size)
            vision: vision modality feature of shape (N, seq. length, vision_input_size)
            labels: labels of shape (N, 1) and ranges from -3 to 3
        '''
        self.text = text
        self.audio = audio
        self.vision = vision
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns an individual data composed of (features, label)
        where features is a dictionary {'text': , 'audio':, 'vision':}
        Returns:
            features['text']: text modality feature of shape (seq. length, text_input_size)
            features['audio']: audio modality feature of shape (audio_input_size)
            features['vision']: vision modality feature of shape (vision_input_size)
            label: a scalar label that ranges from -3 to 3
        '''
        features = dict()
        features['text'] = self.text[idx]
        # audio and vision features are averaged across time
        features['audio'] = np.mean(self.audio[idx], axis=0)
        features['vision'] = np.mean(self.vision[idx], axis=0)
        label = self.labels[idx] > 0

        return features, label