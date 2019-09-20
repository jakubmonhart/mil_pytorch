import mil_pytorch.mil as mil
from mil_pytorch.utils import eval_utils, data_utils, train_utils, create_bags_simple


import numpy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_classification
import time
import os.path

dir_path = os.path.abspath(os.path.dirname(__file__)) + '/'

def create_model(n_neurons1, n_neurons2, n_neurons3):
    # Define neural networks for processing of data before and after aggregation
    prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
        torch.nn.ReLU(),
        # torch.nn.Linear(n_neurons1, n_neurons2, bias = True),
        # torch.nn.ReLU(),
    )

    afterNN1 = torch.nn.Sequential(
        torch.nn.Linear(n_neurons1, 1),
        torch.nn.Tanh()
    )

    # Define model ,loss function and optimizer
    model = torch.nn.Sequential(
        mil.BagModel(prepNN1, afterNN1, torch.mean, device = device)
    ).double()

    return model


# --- CONFIG ---

# Configurations
n_neurons1 = 15
n_neurons2 = 15
n_neurons3 = 15
learning_rate = 1e-4
weight_decay = 1e-4
epochs = 4000
batch_size = 0
patience = 6
delta = 2e-12

print('INFO: CONFIG -')
print('n_neurons1:\t{}\nn_neurons2:\t{}\nn_neurons3:\t{}\nlearning_rate:\t{}\nweight_decay:\t{}\nepochs:\t\t{}\nbatch_size:\t{}\npatience:\t{}\ndelta:\t\t{}'.format(n_neurons1, n_neurons2, n_neurons3, learning_rate, weight_decay, epochs, batch_size, patience, delta))


# --- DATA ---

import pandas

data = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk1/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk1/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk1/labels.csv', sep = '\t', header = None).values.reshape(-1)
data = torch.Tensor(data).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)
print('INFO: Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Move data to gpu (if available)
data = data.double().to(device)
ids = ids.long().to(device)
labels = labels.long().to(device)

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

# Create dataset and divide to train, valid and test part
dataset = mil.MilDataset(data, ids, labels, normalize = True)

train_indices, valid_indices, test_indices = data_utils.data_split(dataset = dataset, valid_ratio = 0.2, test_ratio = 0.2, shuffle = True, stratify = True)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create dataloaders
if batch_size == 0:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = len(train_indices), collate_fn=mil.collate)
else:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = batch_size, collate_fn=mil.collate)

valid_dl = DataLoader(dataset, sampler = valid_sampler, batch_size = len(valid_indices), collate_fn=mil.collate)
test_dl = DataLoader(dataset, sampler = test_sampler, batch_size = len(test_indices), collate_fn=mil.collate)


# --- MODEL ---

model = create_model(n_neurons1, n_neurons2, n_neurons3)
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Move model to gpu if available
model = model.to(device)


# --- TRAIN ---

train_utils.train_model(model, criterion, optimizer, train_dl, valid_dl, epochs, patience, delta)


# --- EVAL ---

eval_utils.evaluation(model, criterion, train_dl, test_dl, device)