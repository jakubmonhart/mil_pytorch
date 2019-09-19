import torch
import numpy
from sklearn import model_selection

def create_3d_data(instances, n_instances):
    ''' 
    Create 3d tensor of data from 2d sequence of instances
    '''
    max_n_instances = max(n_instances)
    n_bags = len(n_instances)
    n_features = instances.shape[1]
    instances = instances.float()
    # Pre-allocate empty 3d tensor
    data = torch.empty(size = (n_bags, max_n_instances, n_features), dtype = torch.double)
    # data = torch.Tensor(n_bags, max_n_instances, n_features)
    
    # n_instances = n_instances.float()

    # Fill data tensor
    marker = 0
    for i in range(n_bags):
        data[i] = torch.cat([ instances[ marker : marker + n_instances[i] ] ,  torch.zeros(max_n_instances - n_instances[i], n_features, dtype = torch.float) ], dim = 0)
        marker += n_instances[i]

    return data



def ids2n_instances(ids):
    unique, inverse, counts = torch.unique(ids, sorted = True, return_inverse = True, return_counts = True)
    idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
    bags = unique[idx]
    counts = counts[idx]
    
    counts = counts.long()

    return counts



def train_test_split_3d(data, labels, n_instances, shuffle, test_size = 0.2):
    data = numpy.array(data)
    labels = numpy.array(labels)
    
    X_train, X_test, y_train, y_test, n_instances, n_instances_t = \
        model_selection.model_selection.train_test_split(data, labels, n_instances, test_size = test_size, shuffle = shuffle)
    
    y = torch.from_numpy(y_train).float()
    x = torch.from_numpy(X_train)
    x_t = torch.from_numpy(X_test)
    y_t = torch.from_numpy(y_test).float()
    n_instances = torch.from_numpy(n_instances).long()
    n_instances_t = torch.from_numpy(n_instances_t).long()
    
    return x, x_t, y, y_t, n_instances, n_instances_t



def data_split(dataset, valid_ratio = 0.2, test_ratio = 0.2, shuffle = True, stratify = True):
    '''
    Splits dataset to test and train data using stratified sampling and subsets
    '''
    indices = numpy.arange(len(dataset))

    if valid_ratio == 0:
        valid_indices = []

        if stratify:
            labels = dataset.labels.cpu()
            train_indices, test_indices = model_selection.train_test_split(indices, stratify = labels, shuffle = shuffle, test_size = test_ratio)
        else:
            train_indices, test_indices = model_selection.train_test_split(indices, stratify = None, shuffle = shuffle, test_size = test_ratio)
    else:
        train_ratio = 1 - (valid_ratio + test_ratio)

        if stratify:
            labels = dataset.labels.cpu()

            train_indices, remain_indices = model_selection.train_test_split(indices, stratify = labels, shuffle = shuffle, test_size = 1 - train_ratio)

            valid_indices, test_indices = model_selection.train_test_split(remain_indices, stratify = labels[remain_indices], shuffle = shuffle, test_size=test_ratio/(test_ratio + valid_ratio))
        else:
            train_indices, remain_indices = model_selection.train_test_split(indices, stratify = None, shuffle = shuffle, test_size = 1 - train_ratio)

            valid_indices, test_indices = model_selection.train_test_split(remain_indices, stratify = labels[remain_indices], shuffle = None, test_size=test_ratio/(test_ratio + valid_ratio))

    return train_indices, valid_indices, test_indices



def create_bag_labels(instance_labels, bagids):
    """
    Creates labels of bags from labels of instances
    """

    bags = torch.unique(bagids)

    n_bags = len(bags)

    # Allocate memory for bag labels
    bag_labels = torch.empty(n_bags, dtype = torch.long)

    for i, bag in enumerate(bags):
        if 1 in instance_labels[bagids == bag]:
            bag_labels[i] = 1
        else:
            bag_labels[i] = 0
    
    return bag_labels