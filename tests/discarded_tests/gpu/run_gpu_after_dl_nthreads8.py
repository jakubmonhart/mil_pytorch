OMP_NUM_THREADS=8

import mill_python.create_dataset.create_bags as create_bags
import mill_python.src.mil_pytorch_gpu as mil
import mill_python.src.utils as utils
import mill_python.train.train as train
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
import numpy
import time

# Configurations
n_neurons1 = 5
# n_neurons2 = 5
# n_neurons3 = 5
learning_rate = 1e-3
weight_decay = 1e-4 
epochs = 2000
pos = 10
neg = 10
class_sep = 1.0

print_loss = True

# Create data
source_data, source_labels = make_classification(n_samples = 100, n_features = 5, n_informative = 5, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)

data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = pos, neg = neg, max_subbags = 5, max_instances = 5)

print("INFO: Data shape:", data.shape, len(labels))

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Don't move data to gpu just yet
data = torch.Tensor(data).double()
ids = torch.Tensor(ids).long()
labels = torch.Tensor(labels).long()

# Convert from (1, 0) to (1, -1)
labels[labels == 0] = -1
# print('DEBUG: labels: {}'.format(labels))

# Dataset
dataset = mil.MilDataset(data, ids, labels, normalize = True)
train_ds, test_ds = mil.train_test_split(dataset, test_size = 0.2)

# Dataloader
train_dl = DataLoader(train_ds, batch_size = len(train_ds), shuffle = True, collate_fn=mil.collate)
# train_batch_dl = DataLoader(train_ds, batch_size = 2, shuffle = True, collate_fn=mil.collate)
test_dl = DataLoader(test_ds, batch_size = len(test_ds), shuffle = False, collate_fn=mil.collate)

# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(len(dataset.data[0]), n_neurons1, bias = True),
    torch.nn.ReLU(),
)

afterNN1 = torch.nn.Sequential(
    torch.nn.Identity()
)

prepNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons1, n_neurons1, bias = True),
    torch.nn.ReLU(),
)

afterNN2 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons1, 1),
    torch.nn.Tanh()
)

# Model and loss function
model = torch.nn.Sequential(
    mil.BagModel(prepNN1, afterNN1, torch.mean, device),
    mil.BagModel(prepNN2, afterNN2, torch.mean, device)
).double()
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Move model to gpu if available
model = model.to(device)

# Train model
start = time.time()

# Parsed fit function
for epoch in range(epochs):
        for data, ids, labels in train_dl:
            # Move data to gpu
            data = data.to(device)
            ids = ids.to(device)
            labels = labels.to(device)

            pred = model((data, ids))
            # print("DEBUG: pred: {} labels: {}".format(pred, labels))
            loss = criterion(pred[:,0], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # Print loss every so epochs
        if print_loss and ((epoch+1)%50 == 0):
            print('step: {:3d} loss: {}'.format(epoch+1, loss))

print('INFO: Finished training - elapsed time: {}'.format(time.time() - start))


# Train set accuracy
for data, ids, labels in train_dl:
    data = data.to(device)
    ids = ids.to(device)
    labels = labels.to(device)

    print("DEBUG: (in train set accuracy) ids: {} data: {}".format(ids, data))
    pred = model((data, ids)) 
    loss = criterion(pred[:,0], labels)

    pred = pred.cpu()
    labels = labels.cpu()

    eer_fpr, eer_fnr = utils.eer(pred[:,0], labels)
    print('loss_train: {}'.format(loss))
    print('acc_train: {}'.format(utils.accuracy(pred[:,0], labels)))
    print('eer_fpr_train: {}'.format(eer_fpr))
    print('eer_fnr_train: {}'.format(eer_fnr))
    
# Test set accuracy
for data, ids, labels in test_dl:
    data = data.to(device)
    ids = ids.to(device)
    labels = labels.to(device)
    pred = model((data, ids))
    loss = criterion(pred[:,0], labels)

    pred = pred.cpu()
    labels = labels.cpu()

    eer_fpr, eer_fnr = utils.eer(pred[:,0], labels)
    print('loss_test: {}'.format(loss))
    print('acc_test: {}'.format(utils.accuracy(pred[:,0], labels)))
    print('eer_fpr_test: {}'.format(eer_fpr))
    print('eer_fnr_test: {}'.format(eer_fnr))
