import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import mean_squared_error

import pickle

import numpy as np

from rank_adaptive_model import CP_tensor_fusion_network

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

def get_kl_loss(model, kl_multiplier, no_kl_epochs, warmup_epochs, epoch):
    '''
    kl loss for rank reduction
    '''
    kl_loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "tensor"):

            kl_loss += layer.tensor.get_kl_divergence_to_prior()
    kl_mult = kl_multiplier * torch.clamp(
                            torch.tensor((
                                (epoch - no_kl_epochs) / warmup_epochs)), 0.0, 1.0)
    """
    print("KL loss ",kl_loss.item())
    print("KL Mult ",kl_mult.item())
    """
    return kl_loss*kl_mult.to(kl_loss.device)


def train_CMU_mosi(batch_size=32, rank_loss=True, kl_multiplier=1.0, no_kl_epochs=5, warmup_epochs=50, epochs=100,
                   max_rank=20, lr=.001):

    # load dataset file
    file = open('mosi_20_seq_data.pkl', 'rb')
    data = pickle.load(file)
    file.close()

    # prepare the datasets and data loaders
    train_set = MultimodalDataset(data['train']['text'], data['train']['audio'],
                                  data['train']['vision'], data['train']['labels'])
    valid_set = MultimodalDataset(data['valid']['text'], data['valid']['audio'],
                                  data['valid']['vision'], data['valid']['labels'])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=len(valid_set))

    # set up model
    input_sizes = (train_set[0][0]['audio'].shape[0], train_set[0][0]['vision'].shape[0],
                   train_set[0][0]['text'].shape[1])
    hidden_sizes = (32, 32, 128)
    output_size = 1

    model = CP_tensor_fusion_network(input_sizes, hidden_sizes, output_size, max_rank)

    # set up training
    DTYPE = torch.FloatTensor
    optimizer = optim.Adam(list(model.parameters()), lr=lr)
    criterion = nn.MSELoss(size_average=False)

    # train and validate
    for e in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            model.zero_grad()

            features, label = batch
            x_a = Variable(features['audio'].float().type(DTYPE), requires_grad=False)
            x_v = Variable(features['vision'].float().type(DTYPE), requires_grad=False)
            x_t = Variable(features['text'].float().type(DTYPE), requires_grad=False)
            y = Variable(label.view(-1, 1).float().type(DTYPE), requires_grad=False)

            output = model([x_a, x_v, x_t])

            loss = criterion(output, y)

            # rank loss for adaptive-rank model
            if rank_loss:
                ard_loss = get_kl_loss(model, kl_multiplier, no_kl_epochs, warmup_epochs, epochs)
                loss += ard_loss

            loss.backward()
            train_loss += loss.item() / len(train_set)

            optimizer.step()

        print("Epoch {}".format(e))
        print("Training Loss {:.2f}".format(train_loss))

        # validate
        model.eval()
        for batch in valid_dataloader:
            features, label = batch
            x_a = Variable(features['audio'].float().type(DTYPE), requires_grad=False)
            x_v = Variable(features['vision'].float().type(DTYPE), requires_grad=False)
            x_t = Variable(features['text'].float().type(DTYPE), requires_grad=False)
            y = Variable(label.view(-1, 1).float().type(DTYPE), requires_grad=False)

            output = model([x_a, x_v, x_t])

        output_valid = output.detach().numpy().reshape(-1)
        y = y.numpy().reshape(-1)

        # validation mean squared error
        valid_mse = mean_squared_error(output_valid, y)
        print("Validation MSE {:.2f}".format(valid_mse))

train_CMU_mosi(kl_multiplier=1e-4)