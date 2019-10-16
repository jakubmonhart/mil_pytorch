import mil_pytorch.mil as mil
from mil_pytorch.utils import eval_utils, data_utils, train_utils, create_bags_simple

import numpy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_classification
import time


def create_model(input_len, n_neurons1, n_neurons2):
    # Define neural networks for processing of data before and after aggregation
    prepNN1 = torch.nn.Sequential(
        torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons1, n_neurons2, bias = True),
        torch.nn.ReLU(),
    )

    afterNN1 = torch.nn.Sequential(
        torch.nn.Linear(n_neurons2, 1),
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
learning_rate = 1e-4
weight_decay = 1e-4
epochs = 1000
pos = 1000
neg = 1000
class_sep = 1.0
n_features = 100
max_instances = 100
batch_size = 0
patience = 20
delta = 0

print('INFO: CONFIG -')
print('n_neurons1:\t{}\nn_neurons2:\t{}\nlearning_rate:\t{}\nweight_decay:\t{}\nepochs:\t\t{}\npos:\t\t{}\nneg:\t\t{}\nclass_sep:\t{}\nn_features:\t{}\nmax_instances:\t{}\nbatch_size:\t{}\npatience:\t{}\ndelta:\t\t{}'.format(n_neurons1, n_neurons2, learning_rate, weight_decay, epochs, pos, neg, class_sep, n_features, max_instances, batch_size, patience, delta))


# --- DATA ---

# Create data
source_data, source_labels = make_classification(n_samples = 5000, n_features = n_features, n_informative = n_features, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)
data, ids, labels = create_bags_simple.create_bags(source_data, source_labels, pos = pos, neg = neg, max_instances = max_instances)
print("INFO: data shape - ", data.shape, len(labels))

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Move data to gpu (if available)
data = data.double().to(device)
ids = ids.long().to(device)
labels = labels.long().to(device)

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

# Create dataset and divide to train and test part
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

model = create_model(len(dataset.data[0]), n_neurons1, n_neurons2)
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Move model to gpu if available
model = model.to(device)


# --- TRAIN ---

train_utils.train_model(model, criterion, optimizer, train_dl, valid_dl, epochs, patience, delta, device = device)
    

# --- EVAL ---

eval_utils.evaluation(model, criterion, train_dl, test_dl, device)
