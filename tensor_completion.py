#%%
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%%

train_batch_size=64

test_batch_size=1000

epochs=14
lr=1.0
gamma=0.7
seed=1
    
torch.manual_seed(seed)

device = torch.device("cuda")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1,batch_size=train_batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, test_batch_size)

#%%

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()


# %%
import tensorly as tl
tl.set_backend('pytorch')
dims = [100,100,10]
rank = 5

#%%

import tensorly.random
import torch

_, factors = tl.random.random_kruskal(dims,rank)
full = tl.kruskal_to_tensor((None,factors))

_,new_factors = tl.random.random_kruskal(dims,rank)

for factor in factors+new_factors:
    print(torch.norm(factor))


new_factors = [torch.tensor(factor,requires_grad=True) for factor in new_factors]

def loss(weights,new_factors):
    pred = tl.kruskal_to_tensor((None,new_factors))  
    return torch.norm(pred-full)/torch.norm(full)

optimizer = torch.optim.SGD(new_factors,lr=0.01)

rank_var = torch.zeros(rank)


#%%

def update_rank_var(rank_var,alpha=1.0):

    M = torch.sum(torch.stack([torch.sum(torch.square(factor),dim=0) for factor in new_factors]),dim=0)

    D = 1.0*torch.sum(torch.tensor(dims))
    prior_type = 'log_uniform'

    if prior_type=='half_cauchy':
        eta = 1.0
        update = (M - D * eta**2 + torch.sqrt(
                    torch.square(M) + (2.0 * D + 8.0) * torch.square(eta) * M +
                    torch.pow(eta, 4.0) * torch.square(D))) / (2.0 * D + 4.0)
    elif prior_type=='log_uniform':
        update = M/D

    rank_var = alpha*rank_var+(1-alpha)*update

    return rank_var


#%%



#TODO
add

#%%

num_steps = 10000

for _ in range(num_steps):

    optimizer.zero_grad()

    loss_value = loss(weights,new_factors)

    loss_value.backward()

    optimizer.step()

    print(loss(weights,new_factors))
# %%
