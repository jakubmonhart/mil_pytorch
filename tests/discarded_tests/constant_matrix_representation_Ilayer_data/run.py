from torch import nn
from torch.nn import functional as F
import torch
from sklearn import metrics
import time
from sklearn.datasets import make_classification
from mil_pytorch.create_dataset.create_bags_simple import create_bags_simple
import numpy
from sklearn import model_selection

import mil_pytorch.src.mil_pytorch_gpu as mil
from mil_pytorch.src import data_utils
from mil_pytorch.src import eval



# Configuration
n_neurons1 = 15
learning_rate = 1e-3
weight_decay = 1e-3
epochs = 4000
pos = 500
neg = 500
max_instances = 50
n_features = 100
class_sep = 1.0

# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(n_features, n_neurons1, bias = True),
    torch.nn.ReLU(),
).double()

afterNN1 = torch.nn.Sequential(
    torch.nn.Linear(n_neurons1, 1, bias = True),
    torch.nn.Tanh()
).double()

# Create data
source_data, source_labels = make_classification(n_samples = 5000, n_features = n_features, n_informative = n_features, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)
data, ids, labels = create_bags_simple(source_data, source_labels, pos = pos, neg = neg, max_instances = max_instances)
n_instances = data_utils.ids2n_instances(torch.Tensor(ids))
data_3d = data_utils.create_3d_data(instances = data, n_instances = n_instances)

print("Data shape:", data_3d.shape, len(labels))

# Convert from (1, 0) to (1, -1)
labels[labels == 0] = -1

# Check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: Using device: {}".format(device))

# Split data on train and test sets
labels = numpy.array(labels)
data_3d = numpy.array(data_3d)
n_instances = numpy.array(n_instances.float())
x, x_t, y, y_t, n_instances, n_instances_t = data_utils.train_test_split_3d(data_3d, labels, n_instances, shuffle = True)

# # Move data to gpu (if available)
x = x.double().to(device)
y = y.long().to(device)
n_instances = n_instances.to(device)

x_t = x_t.double().to(device)
y_t = y_t.long().to(device)
n_instances_t = n_instances_t.to(device)

# Init model
model = mil.BagModel_3d(prepNN1, afterNN1, torch.mean, device = device)
model = model.double()

# Move model to gpu if available
model = model.to(device)

# Criterion and optimizer
criterion = mil.MyHingeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Fit
start = time.time()

print_loss = True
x = x.double()

for epoch in range(epochs):
    pred = model(x, n_instances)
    loss = criterion(pred[:,0], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every ** epochs
    if print_loss and ((epoch+1)%50 == 0):
        print('step: {:3d} loss: {}'.format(epoch+1, loss))

print('INFO | Finished training - elapsed time: {}'.format(time.time() - start))

# Train set accuracy
pred = model(x, n_instances) 
loss = criterion(pred[:,0], y)
pred = pred.cpu()
# y = y.to('cpu')
eer_fpr, eer_fnr = eval.eer(pred[:,0], y)
print('loss_train: {}'.format(loss))
print('acc_train: {}'.format(eval.accuracy(pred[:,0], y)))
print('eer_fpr_train: {}'.format(eer_fpr))
print('eer_fnr_train: {}'.format(eer_fnr))

# Test set accuracy
x_t = x_t.double()
n_instances_t = torch.Tensor(n_instances_t).long()
pred = model(x_t, n_instances_t)
loss = criterion(pred[:,0], y_t)
pred = pred.cpu()
# y_t = y_t.to('cpu')
eer_fpr, eer_fnr = eval.eer(pred[:,0], y_t)
print('loss_test: {}'.format(loss))
print('acc_test: {}'.format(eval.accuracy(pred[:,0], y_t)))
print('eer_fpr_test: {}'.format(eer_fpr))
print('eer_fnr_test: {}'.format(eer_fnr))