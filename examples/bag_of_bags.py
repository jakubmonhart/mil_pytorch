#!/usr/bin/env python

import sys
sys.path.append("../src")
sys.path.append("src")

import mil
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# --- CONFIG ---

# Configurations
n_neurons1 = 15
n_neurons2 = 15
lr = 1e-3
n_epochs = 100
batch_size = 4


# --- DATA ---

# Load data from files
data = pandas.read_csv('data/bag_of_bags/data.csv', header=None).values
ids = pandas.read_csv('data/bag_of_bags/ids.csv', header=None).values
labels = pandas.read_csv('data/bag_of_bags/labels.csv', header=None).values.reshape(-1)

# Create tensors containing data
data = torch.tensor(data, dtype=torch.float).T
ids = torch.tensor(ids)
labels = torch.tensor(labels).float()

labels[labels==-1] = 0

# Create instance of MilDataset
dataset = mil.MilDataset(data, ids, labels, normalize = True)

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Move data to gpu (if available)
data = data.to(device).T
ids = ids.to(device)
labels = labels.to(device)

# Create dataset and divide to train, valid and test part
dataset = mil.MilDataset(data, ids, labels, normalize=True)

train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.labels)
train, test = Subset(dataset, train_indices), Subset(dataset, test_indices)
train_dl, test_dl = DataLoader(train, batch_size=batch_size, collate_fn=mil.collate, drop_last=True), \
                     DataLoader(test, batch_size=batch_size, collate_fn=mil.collate, drop_last=True)


# Defining neural networks for proccesing inputs before and after aggregation function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
    torch.nn.ReLU()
)

afterNN1 = torch.nn.Identity()

prepNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons1, n_neurons2, bias = True),
    torch.nn.ReLU()
)

afterNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons2, 1, bias = True)
)

# Create model, using custom created prepNN, afterNN and aggregation function
model = torch.nn.Sequential( 
    mil.BagModel(prepNN1, afterNN1, aggregation_func = torch.mean),
    mil.BagModel(prepNN2, afterNN2, aggregation_func = torch.mean)
)

# Loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# --- TRAIN ---

losses = []
running_loss = 0.0

for t in range(n_epochs):
  for data, bagids, labels in train_dl:

    pred = model((data, bagids)).squeeze()
    loss = criterion(pred, labels)

    # Optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  # Log
  losses.append(running_loss/len(train_dl))
  running_loss = 0.0
  if (t+1) % 10 == 0:
    print('epoch: {} | loss: {:.3f}'.format(t+1, sum(losses[-10:])/10))


# --- EVAL ---

# Train
correct_count = 0
total_count = 0
for data, bagids, labels in train_dl:
  pred = model((data, bagids)).squeeze() > 0.5
  correct_count += (pred==labels).sum()
  total_count += len(labels)

print('train acc: {:.1f} %'.format((correct_count/total_count)*100))

# Test
correct_count = 0
total_count = 0
for data, bagids, labels in test_dl:
  pred = model((data, bagids)).squeeze() > 0.5
  correct_count += (pred==labels).sum()
  total_count += len(labels)

print('test acc: {:.1f} %'.format((correct_count/total_count)*100))