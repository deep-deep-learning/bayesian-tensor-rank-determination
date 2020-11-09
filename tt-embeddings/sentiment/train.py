#%%


import torch_bayesian_tensor_layers
import torch_bayesian_tensor_layers.layers
from torch_bayesian_tensor_layers.layers import TensorizedEmbedding


#%%
import argparse
import sys
sys.path.insert(0, '..')
import torch_bayesian_tensor_layers

parser = argparse.ArgumentParser()
parser.add_argument(
    '--embedding', 
    default='CP',
    choices=['CP','TensorTrain', 'TensorTrainMatrix', 'Tucker'],
    type=str)
parser.add_argument('--ranks', type=int, default=8)
parser.add_argument('--d', type=int, default=3)
parser.add_argument('--embed_dim', type=int)
parser.add_argument('--voc_dim', default=25000, type=int)
parser.add_argument('--lr', default=5e-4)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--n_epochs',  default=10, type=int)
parser.add_argument('--fout',  default="logdir/", type=str)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument(
    '--dataset',
    default='imdb',
    choices=['imdb', 'sst3', 'sst5'],
    type=str)
args = parser.parse_args()

if args.embedding == 'CP':
    args.ranks = 100
    args.voc_dim= 5*8*25*25
else:             
    args.ranks = 10

model_name = f"{args.dataset}-dim_{args.embed_dim}-ranks_{args.ranks}"


import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import t3nsor as t3
from torchtext import data
from torchtext import datasets
import torch.optim as optim
from models import LSTM_Classifier
from utils import binary_accuracy, train, evaluate
import pickle
import random

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEXT = data.Field(tokenize='spacy', fix_length=1000)
LABEL = data.LabelField(dtype=torch.float)

print('Building dataset...')
if args.dataset == 'imdb':
    OUTPUT_DIM = 1
    train_data, test_ = datasets.IMDB.splits(TEXT, LABEL)
    num_train_examples = len(train_data)
    test_list = list(test_)
    random.shuffle(test_list)
    test_data_ = test_list[:12500]
    val_data_ = test_list[12500:]
    valid_data = data.dataset.Dataset(
        val_data_, fields=[('text', TEXT), ('label', LABEL)])
    test_data = data.dataset.Dataset(
        test_data_, fields=[('text', TEXT), ('label', LABEL)])
elif args.dataset[:3] == 'sst':
    raise ValueError('Need number of train examples')
    OUTPUT_DIM = int(args.dataset[3])
    fine_grained = (OUTPUT_DIM == 5)
    train_data, valid_data, test_data = datasets.SST.splits(
        TEXT, LABEL, fine_grained=fine_grained)
else:
    raise NotImplementedError
print('Done')


def sort_key(ex):
    return len(ex.text)

TEXT.build_vocab(train_data, max_size=args.voc_dim - 2)
LABEL.build_vocab(train_data)

BATCH_SIZE = 256

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


valid_iterator.sort_key = sort_key
test_iterator.sort_key = sort_key

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = args.dropout

actual_vocab_size = len(TEXT.vocab.stoi)


lstm_model = LSTM_Classifier(embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_DIM,
                             output_dim=OUTPUT_DIM,
                             n_layers=N_LAYERS,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)


if args.embedding in ['CP']:

        embed_model = torch_bayesian_tensor_layers.layers.TensorizedEmbedding(
            tensor_type=args.embedding,
            shape = [[5,8,25,25],[4,8,8]],
            max_rank=100,
            padding_idx=1,
            em_stepsize=0.1
        )

        compression_rate = 10.0
    
elif args.embedding in ['TensorTrain','TensorTrainMatrix','Tucker']:
        embed_model = torch_bayesian_tensor_layers.layers.TensorizedEmbedding(
            tensor_type=args.embedding,
            shape = [[5,8,25,25],[16,16]],
            max_rank=20,
            padding_idx=1
        )

        compression_rate = 10.0

else:
    embed_model = nn.Embedding(
        num_embeddings=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM
    )
    compression_rate = 1.0


def cross_entropy_loss(logits, target):
    labels = target.type(torch.LongTensor).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


model = nn.Sequential(embed_model, lstm_model)


if args.dataset == 'imdb':
    criterion = nn.BCEWithLogitsLoss()
    #criterion = criterion.to(device)
elif args.dataset[:3] == 'sst':
    criterion = nn.CrossEntropyLoss()
    #criterion = criterion.to(device)
else:
    raise NotImplementedError

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)
N_EPOCHS = args.n_epochs

log = {
    'compression_rate':compression_rate,
    'train_loss':[], 'test_loss':[], 'valid_loss':[],
    'train_acc':[], 'test_acc':[], 'valid_acc':[]}
best_result = {
    "epoch": 0, "train_acc": 0, "valid_acc": 0, "train_acc": 0}

for epoch in range(N_EPOCHS):


    epoch_multiplier = torch.tensor((epoch-20)/50)
    epoch_multiplier = torch.clamp(epoch_multiplier,0.0,1.0)
    print("Epoch mult",epoch_multiplier)

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion,kl_coeff=5e-3*epoch_multiplier*BATCH_SIZE/num_train_examples)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    log['train_loss'].append(train_loss)
    log['test_loss'].append(test_loss)
    log['train_acc'].append(train_acc)
    log['test_acc'].append(test_acc)
    log['valid_acc'].append(valid_acc)
    log['valid_loss'].append(valid_loss)

    if best_result["valid_acc"] < valid_acc:
        best_result["epoch"] = epoch
        best_result["train_acc"] = train_acc
        best_result["valid_acc"] = valid_acc
        best_result["test_acc"] = test_acc

    if args.fout is not None:
        with open(args.fout+f"{model_name}-best.pkl", 'wb') as f:
            pickle.dump(best_result, f)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
    print ("TEST ACCURACY:", np.round(best_result["test_acc"] * 100, 2))
