import pandas
import numpy
import torch

from mil_pytorch.utils import data_utils



#################
# --- MUSK1 ----#
#################

data = pandas.read_csv('musk1_source/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv('musk1_source/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv('musk1_source/labels.csv', sep = '\t', header = None).values.reshape(-1)

data = torch.Tensor(data).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)
print('INFO: Creating Musk1 data - Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

data_df = pandas.DataFrame(data.numpy())
ids_df = pandas.DataFrame(ids.numpy())
labels_df = pandas.DataFrame(labels.numpy())

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

data_df.to_csv('musk1/data.csv', header = None, index = False)
ids_df.to_csv('musk1/ids.csv', header = None, index = False)
labels_df.to_csv('musk1/labels.csv', header = None, index = False)



#################
# --- MUSK2 ----#
#################

data = pandas.read_csv('musk2_source/data.csv', sep = '\t', header = None).values
ids = pandas.read_csv('musk2_source/bagids.csv', sep = '\t', header = None).values.reshape(-1)
instance_labels = pandas.read_csv('musk2_source/labels.csv', sep = '\t', header = None).values.reshape(-1)

data = torch.Tensor(data).double().t()
ids = torch.Tensor(ids).long()
instance_labels = torch.Tensor(instance_labels).long()
labels = data_utils.create_bag_labels(instance_labels, ids)
print('INFO: Creating Musk 2 data - Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

data_df = pandas.DataFrame(data.numpy())
ids_df = pandas.DataFrame(ids.numpy())
labels_df = pandas.DataFrame(labels.numpy())

# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

data_df.to_csv('musk2/data.csv', header = None, index = False)
ids_df.to_csv('musk2/ids.csv', header = None, index = False)
labels_df.to_csv('musk2/labels.csv', header = None, index = False)



#######################
# --- BAG-OF-BAGS ----#
#######################
pos = 40
neg = 40
n_features = 20
class_sep = 1.5
max_subbags = 7
max_instances = 7

# Create data
from sklearn.datasets import make_classification

source_data, source_labels = make_classification(n_samples = 2000, n_features = n_features, n_informative = n_features, n_redundant = 0, n_repeated = 0, n_classes = 10, class_sep = class_sep, n_clusters_per_class = 1)

from mil_pytorch.utils import create_bags

data, ids, labels = create_bags.create_bags(source_data, source_labels, pos = pos, neg = neg, max_subbags = max_subbags, max_instances = max_instances)
print("INFO: Data shape:", data.shape, len(labels))


# Convert labels from (1, 0) to (1, -1) for tanh
labels[labels == 0] = -1

print('INFO: Creating bag_of_bags data - Data shape -\ndata: {}\nids: {}\nlabels: {}'.format(data.shape, ids.shape, labels.shape))

data = torch.Tensor(data).double()
ids = torch.Tensor(ids).long()
labels = torch.Tensor(labels).long()

data_df = pandas.DataFrame(data.numpy())
ids_df = pandas.DataFrame(ids.numpy())
labels_df = pandas.DataFrame(labels.numpy())

data_df.to_csv('bag_of_bags/data.csv', header = None, index = False)
ids_df.to_csv('bag_of_bags/ids.csv', header = None, index = False)
labels_df.to_csv('bag_of_bags/labels.csv', header = None, index = False)
