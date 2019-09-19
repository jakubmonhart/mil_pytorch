from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
import torch
import numpy
from sklearn import model_selection
# import time



class BagModel(nn.Module):
    '''
    BagModel used with data represented as sequence of instances along with array specifiing number of instances
    Accepts data tensor and n_instances array
    '''

    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()
        
        # TODO - Check for valid dimensions of prepNN and afterNN (with try)

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN
    
    def forward(self, input):    
        ids = input[1]
        input = input[0]
        NN_out = self.prepNN(input) # Forward all instances through neural network
        
        inner_ids = ids[len(ids)-1]
        
        # Numpy version of this segment is faster on CPU (cca 2x - 3x faster .. the differenec is more significatn for longer arrays)
        # start = time.time()
        unique, inverse, counts = torch.unique(inner_ids, sorted = True, return_inverse = True, return_counts = True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]
        # print('Unique | Elapsed time: {}'.format(time.time()-start))

        # Allocate memory for output
        # start = time.time()
        output = torch.empty(len(bags), len(NN_out[0]))
 
        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim = 0)
        
        output = self.afterNN(output.double())
        # print('Aggregation + afterNN | Elapsed time: {}'.format(time.time()-start))

        if (ids.shape[0] == 1):
            return output
        else:
            # start = time.time()
            ids = ids[:len(ids)-1]
            mask = torch.empty(0).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype = torch.int64).reshape(1)))
            # print('Mask for ids | Elapsed time: {}'.format(time.time() - start))
            return (output, ids[:,mask])

class MyHingeLoss(torch.nn.Module):
    '''

    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target):
        target = target.double()
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss<0] = 0
        
        return (torch.sum(hinge_loss) / hinge_loss.size(0))



class MilDataset(Dataset):
    '''

    '''
    def __init__(self, data, ids, labels, normalize = True):
        self.data = data
        self.labels = labels
        self.ids = ids

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))


        self.n_bags = len(torch.unique(ids[0], return_counts = False))

        # Delete constant variables
        self.data = self.data[:, self.data.std(dim = 0) != 0]

        # Normalize
        if normalize:
            std = self.data.std(dim = 0)
            mean = self.data.mean(dim = 0)
            self.data = (self.data - mean)/std
            # print('INFO: data normalized')

    def __len__(self):
        return self.n_bags
    
    def __getitem__(self, index):
        item = self.data[self.ids[0] == index]

        return item, self.ids[:, self.ids[0] == (index)], self.labels[index]
    
    def n_features(self):
        return self.data.size(1)



def collate(batch):
    '''
    Convert to pytorch
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







def train_test_split(dataset, test_size = 0.2, shuffle = True, stratify = True):
    '''
    Splits dataset to test and train data using stratified sampling and subsets
    '''

    indices = numpy.arange(len(dataset))
    labels = dataset.labels.cpu()

    if stratify:
        train_indices, test_indices = model_selection.train_test_split(indices, stratify = labels, shuffle = shuffle, test_size = test_size)
    else:
        train_indices, test_indices = model_selection.train_test_split(indices, stratify = None, shuffle = shuffle, test_size = test_size)

    return [Subset(dataset, train_indices), Subset(dataset, test_indices)]

def max(input, dim):
    result = torch.max(input, dim = dim)
    return result[0]