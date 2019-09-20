import pandas
import numpy
import torch

from mil_pytorch.utils import data_utils

# --- MUSK1 ---

data = pandas.read_csv('musk1_source/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv('musk1_source/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv('musk1_source/labels.csv', sep = '\t', header = None).values.reshape(-1)

data = torch.Tensor(data).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)
print('INFO: Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

data_df = pandas.DataFrame(data.numpy())
ids_df = pandas.DataFrame(ids.numpy())
labels_df = pandas.DataFrame(labels.numpy())

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

data_df.to_csv('musk1/data.csv', header = None, index = False)
ids_df.to_csv('musk1/ids.csv', header = None, index = False)
labels_df.to_csv('musk1/labels.csv', header = None, index = False)

# --- MUSK2 ----

data = pandas.read_csv('musk2_source/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv('musk2_source/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv('musk2_source/labels.csv', sep = '\t', header = None).values.reshape(-1)

data = torch.Tensor(data).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)
print('INFO: Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

data_df = pandas.DataFrame(data.numpy())
ids_df = pandas.DataFrame(ids.numpy())
labels_df = pandas.DataFrame(labels.numpy())

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

data_df.to_csv('musk2/data.csv', header = None, index = False)
ids_df.to_csv('musk2/ids.csv', header = None, index = False)
labels_df.to_csv('musk2/labels.csv', header = None, index = False)
