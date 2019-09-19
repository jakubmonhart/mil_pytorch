import mil_pytorch.mil as mil
from mil_pytorch.utils import eval_utils, data_utils, train_utils, create_bags_simple


import numpy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_classification
import time
import os.path

dir_path = os.path.abspath(os.path.dirname(__file__)) + '/'

def create_model(input_len, n_neurons1, n_neurons2, n_neurons3):
    # Define neural networks for processing of data before and after aggregation
    prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(input_len, n_neurons1, bias = True),
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
        mil.BagModel_3d(prepNN1, afterNN1, torch.mean, device = device)
    ).double()

    # Move model to gpu if available
    model = model.to(device)

    return model

# --- CONFIG ---

# Configurations
n_neurons1 = 15
n_neurons2 = 15
n_neurons3 = 15
learning_rate = 1e-3
weight_decay = 1e-3
epochs = 4000
pos = 50
neg = 50
class_sep = 1.0
n_features = 10
max_instances = 15
batch_size = 0
patience = 20
delta = 0

print('INFO: CONFIG -')
print('n_neurons1:\t{}\nn_neurons2:\t{}\nn_neurons3:\t{}\nlearning_rate:\t{}\nweight_decay:\t{}\nepochs:\t\t{}\npos:\t\t{}\nneg:\t\t{}\nclass_sep:\t{}\nn_features:\t{}\nmax_instances:\t{}\nbatch_size:\t{}\npatience:\t{}\ndelta:\t\t{}'.format(n_neurons1, n_neurons2, n_neurons3, learning_rate, weight_decay, epochs, pos, neg, class_sep, n_features, max_instances, batch_size, patience, delta))

# --- DATA ---

import pandas

instances = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk2/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk2/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv(filepath_or_buffer = dir_path + 'Musk2/labels.csv', sep = '\t', header = None).values.reshape(-1)
instances = torch.Tensor(instances).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)

print("INFO: data shape - ", instances.shape, len(labels))

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Move data to gpu (if available)
instances = instances.double().to(device)
ids = ids.to(device)
labels = labels.long().to(device)

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

# Create dataset and divide to train and test part
dataset = mil.MilDataset_3d(instances, ids, labels, normalize = True)

train_indices, valid_indices, test_indices = data_utils.data_split(dataset = dataset, valid_ratio = 0.2, test_ratio = 0.2, shuffle = True, stratify = True)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create dataloaders
if batch_size == 0:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = len(train_indices))
else:
    train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = batch_size)

valid_dl = DataLoader(dataset, sampler = valid_sampler, batch_size = len(valid_indices))
test_dl = DataLoader(dataset, sampler = test_sampler, batch_size = len(test_indices))

# --- MODEL ---

model = create_model(len(dataset.data[0][0]) , n_neurons1, n_neurons2, n_neurons3)
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)


# --- TRAIN ---

train_utils.train_model(model, criterion, optimizer, train_dl, valid_dl, epochs, patience, delta)

# --- EVAL ---

eval_utils.evaluation(model, criterion, train_dl, test_dl, device)

# --- SAVE LOG ---


