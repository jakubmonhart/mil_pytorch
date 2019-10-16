from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
import torch
import numpy
from sklearn import model_selection

from mil_pytorch.utils import data_utils

class BagModel(nn.Module):
    '''
    Model for solving MIL problems

    Args:
        prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
        afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
        aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used
        device: use 'cuda' when using gpu for forward to move created tensors to gpu.
            Default: 'cpu'

    Returns:
        Output of forward function.
    '''

    def __init__(self, prepNN, afterNN, aggregation_func, device = 'cpu'):
        super().__init__()

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN
        self.device = device
    
    def forward(self, input):    
        ids = input[1]
        input = input[0]
        inner_ids = ids[len(ids)-1]

        NN_out = self.prepNN(input)
            
        unique, inverse, counts = torch.unique(inner_ids, sorted = True, return_inverse = True, return_counts = True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((len(bags), len(NN_out[0])), device = self.device)
 
        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim = 0)
        
        output = self.afterNN(output.double())

        if (ids.shape[0] == 1):
            return output
        else:
            ids = ids[:len(ids)-1]
            mask = torch.empty(0, device = self.device).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype = torch.int64).reshape(1)))
            return (output, ids[:,mask])



class BagModel_3d(nn.Module):
    '''
    Model for solving MIL problems.
    Represents MIL data with variable number of instances in bag with 3d tensor with fixed dimensions and n_instances tensor specifying valid instances in each bag (slice) of this 3d tensor.

    Args:
        prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
        afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
        aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used
        device: use 'cuda' when using gpu for forward to move created tensors to gpu.
            Default: 'cpu'

    Returns:
        Output of forward function.
    '''
    def __init__(self, prepNN, afterNN, aggregation_func, device = 'cpu'):
        super().__init__()
        
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.aggregation_func = aggregation_func
        self.device = device

    def forward(self, input):
        n_instances = input[1]
        input = input[0]

        NN_out = self.prepNN(input)

        output = torch.empty(size = (input.size(0), NN_out.size(2)), dtype = torch.double).to(self.device)

        for i, n in enumerate(n_instances):
            output[i] = self.aggregation_func(NN_out[i, :n], dim = 0)

        output = self.afterNN(output)

        
        return output



class MyHingeLoss(torch.nn.Module):
    '''

    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target):
        target = target.double()
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss<0] = 0
        
        return (torch.sum(hinge_loss, dim = 0, keepdim = True) / hinge_loss.size(0))



class MilDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset. 

    Args:
        data:
        ids:
        labels:
        normalize:
    '''
    def __init__(self, data, ids, labels, normalize = True):
        self.data = data
        self.labels = labels
        self.ids = ids

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        unique, inverse = torch.unique(self.ids[0], sorted = True, return_inverse = True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        self.bags = unique[idx]
        self.n_bags = len(self.bags)

        # Delete constant variables
        self.data = self.data[:, self.data.std(dim = 0) != 0]

        # Normalize
        if normalize:
            std = self.data.std(dim = 0)
            mean = self.data.mean(dim = 0)
            self.data = (self.data - mean)/std

    def __len__(self):
        return self.n_bags
    
    def __getitem__(self, index):
        item = self.data[self.ids[0] == self.bags[index]]

        return item, self.ids[:, self.ids[0] == self.bags[index]], self.labels[index]
    
    def n_features(self):
        return self.data.size(1)



class MilDataset_3d(Dataset):
    '''

    '''

    def __init__(self, instances, ids, labels, normalize = True, device = None):

        # Normalize
        if normalize:
            std = instances.std(dim = 0)
            mean = instances.mean(dim = 0)
            instances = (instances - mean)/std
        
        self.n_instances = data_utils.ids2n_instances(ids)
        self.data = data_utils.create_3d_data(instances, self.n_instances)
        self.labels = labels
        self.n_bags = torch.tensor(len(self.n_instances), dtype = torch.long)

        if device is not None:
            self.data = self.data.to(device)
            self.n_instances = self.n_instances.to(device)
            self.labels = self.labels.to(device)
            self.n_bags = self.n_bags.to(device)


    def __len__(self):
        return self.n_bags
    
    def __getitem__(self, index):
        return self.data[index], self.n_instances[index], self.labels[index]



def collate(batch):
    '''

    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []
    
    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
    
    out_data = torch.cat(batch_data, dim = 0)
    out_bagids = torch.cat(batch_bagids, dim = 1)
    out_labels = torch.stack(batch_labels)
    
    return out_data, out_bagids, out_labels



def collate_np(batch):
    '''

    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []
    
    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
    
    out_data = torch.cat(batch_data, dim = 0)
    out_bagids = torch.cat(batch_bagids, dim = 1)
    out_labels = torch.tensor(batch_labels)
    
    
    return out_data, out_bagids, out_labels


def max(input, dim):
    result = torch.max(input, dim = dim)
    return result[0]